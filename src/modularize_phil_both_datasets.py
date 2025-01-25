import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import datetime
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import time
from functools import partial
import logging
import yaml
import argparse

class SlidingWindowDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=512, stride=256, max_tokens=None, logger=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.max_tokens = max_tokens
        self.logger = logger or logging.getLogger(__name__)
        
        # Load and process text
        self.logger.info(f"Loading data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text
        self.logger.info("Tokenizing text...")
        self.all_tokens = self.tokenizer.encode(text)
        
        # Store total number of tokens before any limiting
        self.total_available_tokens = len(self.all_tokens)
        self.logger.info(f"Total available tokens in {file_path}: {self.total_available_tokens:,}")
        
        # Apply token limit if specified
        if max_tokens is not None:
            if max_tokens > self.total_available_tokens:
                warning_msg = (
                    f"Requested {max_tokens:,} tokens but only {self.total_available_tokens:,} "
                    f"are available in {file_path}. Using all available tokens."
                )
                warnings.warn(warning_msg)
                self.logger.warning(warning_msg)
            else:
                self.logger.info(f"Limiting dataset to {max_tokens:,} tokens")
                self.all_tokens = self.all_tokens[:max_tokens]
        
        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.all_tokens) - seq_length) // stride + 1)
        
        # Log dataset statistics
        stats = {
            "file_path": file_path,
            "total_tokens_after_limiting": len(self.all_tokens),
            "n_sequences": self.n_sequences,
            "sequence_length": seq_length,
            "stride": stride,
            "max_tokens_requested": max_tokens,
            "total_available_tokens": self.total_available_tokens
        }
        
        self.logger.info("Dataset statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")

        # Create examples
        self.examples = []
        for i in range(self.n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.seq_length
            
            sequence = self.all_tokens[start_idx:end_idx]
            attention_mask = [1] * len(sequence)
            
            if len(sequence) < self.seq_length:
                padding_length = self.seq_length - len(sequence)
                sequence = sequence + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            input_ids = torch.tensor(sequence)
            attention_mask = torch.tensor(attention_mask)
            
            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })

        self.logger.info(f"Created {len(self.examples)} examples from {len(self.all_tokens)} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}



class KDRecipeSingleDevice:
    def __init__(self, cfg, logger=None, run_dir=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Use provided run_dir or create new one
        if run_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(cfg['output_dir'], f"run_{timestamp}")
        self.run_dir = run_dir
        
        # Create necessary directories
        self.eval_dir = os.path.join(self.run_dir, "evaluations")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.logger.info(f"Initialized training in directory: {self.run_dir}")
        
        self.log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self.log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        
        self.seed = self._set_seed(cfg['seed'])
        self.epochs_run = 0
        self.total_epochs = cfg['epochs']
        self.max_steps_per_epoch = cfg['max_steps_per_epoch']
        self.global_step = 0
        self.resume_from_checkpoint = cfg['resume_from_checkpoint']
        self.save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.clip_grad_norm = cfg.get("clip_grad_norm", None)
        self.kd_ratio = cfg.get("kd_ratio", 0.5)

        self.eval_every = cfg.get('eval_every', 100)
        self.eval_steps = cfg.get('eval_steps', 100)
        self.train_losses = []
        self.eval_losses = []
        self.train_ppls = []
        self.eval_ppls = []
        self.eval_steps_done = 0  # Add this line

        self.best_val_loss = float('inf')
        self.use_wandb = cfg.get('wandb', {}).get('enabled', False)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.student_model = self._setup_student_model()
        self.teacher_model = self._setup_teacher_model()

        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.cfg['learning_rate'])
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self.steps_per_epoch:
            self.steps_per_epoch = self.max_steps_per_epoch

        self.lr_scheduler = self._setup_lr_scheduler()

        self.scaler = torch.cuda.amp.GradScaler()

        self.ntp_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.kd_loss_fn = self.kl_div_loss

    def _setup_student_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            torch_dtype=self.dtype,
        )
        if self.cfg.get('use_gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        return model.to(self.device)

    def _setup_teacher_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        return model

    def _setup_data(self):
        """Setup data loaders with multiple datasets."""
        self.logger.info("Setting up data loaders...")
        
        # Create datasets
        datasets = []
        total_tokens_requested = 0
        total_tokens_available = 0
        
        for source in self.cfg['data']['sources']:
            dataset = SlidingWindowDataset(
                file_path=source['path'],
                tokenizer=self.tokenizer,
                seq_length=self.cfg['data']['max_length'],
                stride=self.cfg['data']['stride'],
                max_tokens=source['max_tokens'],
                logger=self.logger
            )
            datasets.append(dataset)
            total_tokens_requested += source['max_tokens'] if source['max_tokens'] else dataset.total_available_tokens
            total_tokens_available += dataset.total_available_tokens
            
        # Log total token statistics
        self.logger.info(f"Total tokens requested across all sources: {total_tokens_requested:,}")
        self.logger.info(f"Total tokens available across all sources: {total_tokens_available:,}")

        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        self.logger.info(f"Combined dataset size: {len(combined_dataset)} sequences")

        # Calculate splits
        total_sequences = len(combined_dataset)
        train_size = int(0.8 * total_sequences)
        val_size = int(0.1 * total_sequences)
        test_size = total_sequences - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.logger.info(f"Train size: {train_size:,} sequences")
        self.logger.info(f"Val size: {val_size:,} sequences")
        self.logger.info(f"Test size: {test_size:,} sequences")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg['data']['batch_size'],
            shuffle=True,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg['data']['batch_size'],
            shuffle=False,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg['data']['batch_size'],
            shuffle=False,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader

    def _setup_lr_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_epochs * self.steps_per_epoch),
            num_training_steps=self.total_epochs * self.steps_per_epoch
        )

    def kl_div_loss(self, student_logits, teacher_logits, labels):
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='batchmean'
        )
        return loss

    def _loss_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        shifted_logits = student_logits[..., :-1, :].contiguous()
        shifted_labels = input_ids[..., 1:].contiguous()
        
        ntp_loss = self.ntp_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        kd_loss = self.kd_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), 
                                  teacher_logits[..., :-1, :].contiguous().view(-1, teacher_logits.size(-1)), 
                                  shifted_labels.view(-1))

        # loss = (1 - self.kd_ratio) * ntp_loss + self.kd_ratio * kd_loss
        loss = kd_loss 

        return loss, ntp_loss, kd_loss

    def evaluate(self, eval_loader, steps=None):
        self.student_model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if steps is not None and i >= steps:
                    break
                
                loss, _, _ = self._loss_step(batch)
                total_loss += loss.item()
                total_tokens += batch['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()

        avg_loss = total_loss / (i + 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        if self.use_wandb:
            import wandb
            wandb.log({
                "eval/loss": avg_loss,
                "eval/ppl": ppl,
            }, step=self.global_step)
        
        return avg_loss, ppl

    def generate_samples(self, batch, num_samples=5):
        """Generate and compare predictions from teacher and student."""
        self.student_model.eval()
        self.teacher_model.eval()
        
        # Select a subset of the batch for sampling
        input_ids = batch['input_ids'][:num_samples].to(self.device)
        
        # Get the actual next tokens (targets)
        actual_next = input_ids[:, -1].cpu()
        context = input_ids[:, :-1]
        
        # Generate from both models
        with torch.no_grad():
            student_logits = self.student_model(context).logits[:, -1, :]
            teacher_logits = self.teacher_model(context).logits[:, -1, :]
            
            student_preds = torch.argmax(student_logits, dim=-1).cpu()
            teacher_preds = torch.argmax(teacher_logits, dim=-1).cpu()
        
        # Decode all tokens
        samples = []
        for i in range(num_samples):
            context_text = self.tokenizer.decode(context[i])
            actual_next_text = self.tokenizer.decode(actual_next[i])
            student_pred_text = self.tokenizer.decode(student_preds[i])
            teacher_pred_text = self.tokenizer.decode(teacher_preds[i])
            
            sample = {
                'context': context_text,
                'actual_next': actual_next_text,
                'student_prediction': student_pred_text,
                'teacher_prediction': teacher_pred_text,
            }
            samples.append(sample)
        
        return samples

    def save_samples(self, samples, epoch, step):
        """Save generated samples and log to wandb if enabled."""
        # Create a formatted table for logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.eval_dir, f"samples_epoch{epoch}_step{step}_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"Sample {i}:\n")
                f.write(f"Context: {sample['context']}\n")
                f.write(f"Actual next token: {sample['actual_next']}\n")
                f.write(f"Student prediction: {sample['student_prediction']}\n")
                f.write(f"Teacher prediction: {sample['teacher_prediction']}\n")
                f.write("-" * 80 + "\n")
        
        if self.use_wandb:
            import wandb
            # Create a wandb Table
            columns = ["Sample", "Context", "Actual", "Student", "Teacher"]
            data = [
                [i+1, 
                 sample['context'], 
                 sample['actual_next'],
                 sample['student_prediction'],
                 sample['teacher_prediction']]
                for i, sample in enumerate(samples)
            ]
            
            table = wandb.Table(columns=columns, data=data)
            wandb.log({
                "predictions/examples": table,
                "predictions/epoch": epoch,
                "predictions/step": step
            }, step=self.global_step)

    def train(self):
        self.logger.info("Starting training")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model dtype: {self.dtype}")
        
        for epoch in range(self.epochs_run, self.total_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.total_epochs}")
            
            self.student_model.train()
            total_loss = 0
            total_kd_loss = 0
            total_ntp_loss = 0
            logged_steps = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                if step // self.gradient_accumulation_steps == self.max_steps_per_epoch:
                    break

                with torch.cuda.amp.autocast():
                    loss, ntp_loss, kd_loss = self._loss_step(batch)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.student_model.parameters(), self.clip_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                    total_loss += loss.item() * self.gradient_accumulation_steps
                    total_ntp_loss += ntp_loss.item()
                    total_kd_loss += kd_loss.item()
                    logged_steps += 1

                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

                    if self.global_step % self.log_every_n_steps == 0:
                        self._log_metrics(loss.item() * self.gradient_accumulation_steps, 
                                          ntp_loss.item(), 
                                          kd_loss.item(), 
                                          self.optimizer.param_groups[0]['lr'])

                    if self.global_step % self.eval_every == 0:
                        eval_loss, eval_ppl = self.evaluate(self.val_loader, steps=self.eval_steps)
                        train_loss = total_loss / logged_steps
                        train_ppl = torch.exp(torch.tensor(train_loss)).item()
                        
                        self.train_losses.append(train_loss)
                        self.eval_losses.append(eval_loss)
                        self.train_ppls.append(train_ppl)
                        self.eval_ppls.append(eval_ppl)
                        self.eval_steps_done += 1
                        
                        print(f"Step {self.global_step}: Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}, "
                              f"Eval Loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.4f}")
                        
                        self.plot_metrics()

                        # Generate and save sample predictions
                        samples = self.generate_samples(batch)
                        self.save_samples(samples, epoch, self.global_step)

                        # Log metrics
                        metrics = {
                            "train/loss": train_loss,
                            "train/ppl": train_ppl,
                            "train/kd_loss": total_kd_loss / logged_steps,
                            "train/ntp_loss": total_ntp_loss / logged_steps,
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch + step / len(self.train_loader),
                        }
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log(metrics, step=self.global_step)

                    self.global_step += 1

            pbar.close()
            
            avg_loss = total_loss / self.steps_per_epoch
            avg_ntp_loss = total_ntp_loss / self.steps_per_epoch
            avg_kd_loss = total_kd_loss / self.steps_per_epoch
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg NTP Loss: {avg_ntp_loss:.4f}, Avg KD Loss: {avg_kd_loss:.4f}")
            
            self.epochs_run += 1

            # Evaluate on validation set
            val_loss, val_ppl = self.evaluate(self.val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Validation PPL: {val_ppl:.4f}")

            # Save only if it's the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, avg_loss, val_loss, is_best=True)

            # Save final model at the last epoch
            if epoch == self.total_epochs - 1:
                self.save_checkpoint(epoch, avg_loss, val_loss, is_final=True)

            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1} completed - "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val PPL: {val_ppl:.4f}"
            )

        self.logger.info("Training completed")

    def _log_metrics(self, loss, ntp_loss, kd_loss, lr):
        print(f"Step {self.global_step}: Loss: {loss:.4f}, NTP Loss: {ntp_loss:.4f}, KD Loss: {kd_loss:.4f}, LR: {lr:.6f}")
        if self.log_peak_memory_stats and torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    def plot_metrics(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        x = range(1, self.eval_steps_done + 1)
        plt.plot(x, self.train_losses, label='Train Loss')
        plt.plot(x, self.eval_losses, label='Eval Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(2, 1, 2)
        plt.plot(x, self.train_ppls, label='Train PPL')
        plt.plot(x, self.eval_ppls, label='Eval PPL')
        plt.legend()
        plt.title('Perplexity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'metrics_plot.png'))
        plt.close()

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False, is_final=False):
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'train_ppls': self.train_ppls,
            'eval_ppls': self.eval_ppls,
        }
        
        # Save best model if enabled and this is the best so far
        if is_best and self.cfg['checkpointing']['save_best']:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            self.logger.info(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save(checkpoint, checkpoint_path)
        
        # Save final model if enabled and this is the final epoch
        if is_final and self.cfg['checkpointing']['save_last']:
            checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pt")
            self.logger.info(f"Saving final model at epoch {epoch+1}")
            torch.save(checkpoint, checkpoint_path)
        
        # Save periodic checkpoint if enabled
        save_every_n = self.cfg['checkpointing'].get('save_every_n_epochs')
        if save_every_n and (epoch + 1) % save_every_n == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            self.logger.info(f"Saving periodic checkpoint at epoch {epoch+1}")
            torch.save(checkpoint, checkpoint_path)
            
            # Manage number of checkpoints to keep
            keep_n = self.cfg['checkpointing'].get('keep_n_checkpoints', 3)
            if keep_n:
                checkpoints = sorted(
                    [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_epoch_")],
                    key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                for old_ckpt in checkpoints[:-keep_n]:
                    old_ckpt_path = os.path.join(self.checkpoint_dir, old_ckpt)
                    os.remove(old_ckpt_path)
                    self.logger.info(f"Removed old checkpoint: {old_ckpt}")

        if self.use_wandb:
            import wandb
            # Log best model to wandb
            if is_best:
                wandb.save(os.path.join(self.checkpoint_dir, "best_model.pt"))
            # Log final model to wandb
            if is_final:
                wandb.save(os.path.join(self.checkpoint_dir, "final_model.pt"))

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.epochs_run = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval_losses = checkpoint.get('eval_losses', [])
        self.train_ppls = checkpoint.get('train_ppls', [])
        self.eval_ppls = checkpoint.get('eval_ppls', [])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

def setup_logging(run_dir):
    # Create logs directory
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add our handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Override resume checkpoint path')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)
    
    # Initialize wandb if enabled
    if cfg.get('wandb', {}).get('enabled', False):
        import wandb
        run_name = cfg['wandb'].get('name') or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=cfg['wandb']['project'],
            name=run_name,
            config=cfg,
            tags=cfg['wandb'].get('tags', []),
            notes=cfg['wandb'].get('notes', ''),
        )
        
    # Override resume checkpoint if provided
    if args.resume:
        cfg['checkpointing']['resume'] = args.resume

    # Create unique run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg['output']['dir'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config copy in run directory
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f)

    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(f"Config saved to: {config_save_path}")
    logger.info("Starting new training run with configuration:")
    logger.info(yaml.dump(cfg))

    # Create recipe
    recipe = KDRecipeSingleDevice(cfg, logger=logger, run_dir=run_dir)
    
    try:
        recipe.setup()
        logger.info("Setup completed successfully")
        
        # Resume from checkpoint if specified
        if cfg['checkpointing']['resume']:
            recipe.load_checkpoint(cfg['checkpointing']['resume'])
            logger.info(f"Resumed from checkpoint: {cfg['checkpointing']['resume']}")
        
        recipe.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    finally:
        if cfg.get('wandb', {}).get('enabled', False):
            import wandb
            wandb.finish()

if __name__ == "__main__":
    main()
