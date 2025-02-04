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
import numpy as np
import mmap
from pathlib import Path

class TokenizedDataset(Dataset):
    """Memory efficient dataset that caches chunks in RAM."""
    
    def __init__(self, file_path, tokenizer, seq_length=512, stride=256, max_tokens=None, logger=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.logger = logger or logging.getLogger(__name__)
        
        # Create cache directory
        self.data_path = Path(file_path)
        self.cache_dir = self.data_path.parent / f"{self.data_path.stem}_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_path = self.cache_dir / "tokens.npy"
        
        # Load or create cached tokens
        if self.cache_path.exists():
            self.logger.info(f"Loading cached tokens from {self.cache_path}")
            self.all_tokens = np.load(str(self.cache_path))
        else:
            self.logger.info(f"Creating token cache for {file_path}")
            # Load and tokenize text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = text.replace('\r\n', '\n').replace('\r', '\n')
                tokens = self.tokenizer.encode(text)
                if max_tokens:
                    tokens = tokens[:max_tokens]
                self.all_tokens = np.array(tokens, dtype=np.int32)
            # Save cache
            np.save(str(self.cache_path), self.all_tokens)
        
        self.n_sequences = max(0, (len(self.all_tokens) - seq_length) // stride + 1)
        total_size_gb = self.all_tokens.nbytes / 1024**3
        self.logger.info(f"Dataset contains {len(self.all_tokens):,} tokens ({total_size_gb:.2f} GB)")
        self.logger.info(f"Created {self.n_sequences:,} sequences with stride {stride}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length
        
        # Get sequence
        sequence = self.all_tokens[start_idx:end_idx].copy()
        attention_mask = np.ones(len(sequence), dtype=np.int32)
        
        # Handle padding if needed
        if len(sequence) < self.seq_length:
            padding_length = self.seq_length - len(sequence)
            sequence = np.pad(sequence, (0, padding_length), 
                            constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, padding_length),
                                  constant_values=0)
        
        return {
            'input_ids': torch.from_numpy(sequence).long(),
            'attention_mask': torch.from_numpy(attention_mask).long()
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}



class KDRecipeSingleDevice:
    def __init__(self, cfg, logger=None, run_dir=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Change dtype to float16 for AMP or disable AMP for bfloat16
        if True:
            self.dtype = torch.bfloat16
            self.use_amp = False  # Disable AMP when using bfloat16
        else:
            self.dtype = torch.float16
            self.use_amp = True   # Use AMP with float16
        
        self.logger.info(f"Using dtype: {self.dtype}, AMP enabled: {self.use_amp}")
        
        # Use provided run_dir or create new one
        if run_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(cfg['output']['dir'], f"run_{timestamp}")
        self.run_dir = run_dir
        
        # Create necessary directories
        self.eval_dir = os.path.join(self.run_dir, "evaluations")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.logger.info(f"Initialized training in directory: {self.run_dir}")
        
        self.log_every_n_steps = cfg['training'].get('log_every_n_steps', 100)
        self.log_peak_memory_stats = cfg.get('log_peak_memory_stats', False)
        
        self.seed = self._set_seed(cfg['training']['seed'])
        self.epochs_run = 0
        self.total_epochs = cfg['training']['epochs']
        self.max_steps_per_epoch = cfg['training']['max_steps_per_epoch']
        self.global_step = 0
        self.resume_from_checkpoint = cfg['checkpointing']['resume']
        self.save_adapter_weights_only = cfg.get('save_adapter_weights_only', False)
        self.gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps']
        self.clip_grad_norm = cfg['training'].get('clip_grad_norm', None)
        self.kd_ratio = cfg['training'].get('kd_ratio', 1.0)

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
        self._setup_tokenizer()
        self.student_model = self._setup_student_model()
        self.teacher_model = self._setup_teacher_model()

        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(), 
            lr=self.cfg['training']['learning_rate']
        )
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self.steps_per_epoch:
            self.steps_per_epoch = self.max_steps_per_epoch

        self.lr_scheduler = self._setup_lr_scheduler()

        self.ntp_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.kd_loss_fn = self.kl_div_loss

    def _setup_tokenizer(self):
        """Setup and test tokenizer behavior with special characters."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model']['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Test tokenizer behavior with special characters
        test_cases = [
            "Hello\r\nWorld",  # Windows newline
            "Hello\nWorld",    # Unix newline
            "Hello\rWorld",    # Old Mac newline
            "Hello  World",    # Multiple spaces
            "Hello\n\nWorld",  # Multiple newlines
        ]
        
        self.logger.info("Testing LLaMA tokenizer behavior with special characters:")
        for test in test_cases:
            tokens = self.tokenizer.encode(test)
            decoded = self.tokenizer.decode(tokens)
            self.logger.info(f"\nOriginal: {repr(test)}")
            self.logger.info(f"Decoded:  {repr(decoded)}")
        
        return self.tokenizer

    def _setup_student_model(self):
        """Setup student model - same architecture as teacher but randomly initialized."""
        config = LlamaConfig.from_pretrained(self.cfg['model']['name'])
        model = LlamaForCausalLM(config)  # Random initialization
        
        if self.cfg['model'].get('use_gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        return model.to(self.device)

    def _setup_teacher_model(self):
        """Setup teacher model with optimized quantization."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
            bnb_4bit_use_double_quant=False,  # Disable double quantization
            bnb_4bit_quant_type="nf4",
        )
        
        self.logger.info("Loading teacher model with 4-bit quantization...")
        model = LlamaForCausalLM.from_pretrained(
            self.cfg['model']['name'],
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Optimize memory usage
        model.eval()
        model.requires_grad_(False)  # More efficient than looping through parameters
        
        # Quick verification
        if self.cfg.get('model', {}).get('verify_teacher', True):
            self._verify_teacher_model(model)
        
        return model

    def _verify_teacher_model(self, model):
        """Verify teacher model with minimal memory usage."""
        test_input = self.tokenizer("Test input:", return_tensors="pt").to(self.device)
        with torch.no_grad():
            test_output = model.generate(
                test_input.input_ids,
                max_new_tokens=5,
                num_beams=1,  # Reduce beam size for verification
                temperature=0.7
            )
        test_text = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
        self.logger.info(f"Teacher verification output: {test_text}")

    def _setup_data(self):
        """Setup data loaders with memory-mapped datasets."""
        self.logger.info("Setting up data loaders...")
        
        # Create datasets
        datasets = []
        for source in self.cfg['data']['sources']:
            dataset = TokenizedDataset(
                file_path=source['path'],
                tokenizer=self.tokenizer,
                seq_length=self.cfg['data']['max_length'],
                stride=self.cfg['data']['stride'],
                max_tokens=source.get('max_tokens', None),
                logger=self.logger
            )
            datasets.append(dataset)
        
        # Combine datasets if multiple
        if len(datasets) > 1:
            train_dataset = ConcatDataset(datasets)
            self.logger.info(f"Combined {len(datasets)} datasets:")
        else:
            train_dataset = datasets[0]
        
        # Create test dataset
        test_dataset = TokenizedDataset(
            file_path=self.cfg['data']['test_path'],
            tokenizer=self.tokenizer,
            seq_length=self.cfg['data']['max_length'],
            stride=self.cfg['data']['stride'],
            max_tokens=None,  # No token limit for test set
            logger=self.logger
        )
        
        # Log dataset sizes
        self.logger.info(f"Train dataset size: {len(train_dataset):,} sequences")
        self.logger.info(f"Test dataset size: {len(test_dataset):,} sequences")
        
        # Split train into train and validation
        total_train = len(train_dataset)
        train_size = int(0.9 * total_train)  # 90% for training
        val_size = total_train - train_size   # 10% for validation
        
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.logger.info(f"Final split sizes:")
        self.logger.info(f"Train size: {train_size:,} sequences")
        self.logger.info(f"Val size: {val_size:,} sequences")
        self.logger.info(f"Test size: {len(test_dataset):,} sequences")
        
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
        """Compute loss with careful memory management."""
        # Move batch to device and ensure contiguous memory
        input_ids = batch['input_ids'].to(self.device, non_blocking=True).contiguous()
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True).contiguous()
        
        # Get teacher predictions
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False  # Disable KV cache to save memory
            )
            teacher_logits = teacher_outputs.logits.detach()
            del teacher_outputs
        
        # Get student predictions
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            student_logits = student_outputs.logits
            del student_outputs
        
        # Prepare shifted sequences for loss computation
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Shift and prepare logits/labels
            shifted_logits = student_logits[..., :-1, :].contiguous()
            shifted_labels = input_ids[..., 1:].contiguous()
            shifted_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            
            del student_logits, teacher_logits  # Free original logits
            
            # Compute losses
            ntp_loss = self.ntp_loss_fn(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            )
            
            kd_loss = self.kd_loss_fn(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_teacher_logits.view(-1, shifted_teacher_logits.size(-1)),
                shifted_labels.view(-1)
            )
            
            # Compute weighted loss
            loss = (1 - self.kd_ratio) * ntp_loss + self.kd_ratio * kd_loss
            
            # Clean up remaining tensors
            del shifted_logits, shifted_teacher_logits, shifted_labels
        
        # Force garbage collection and clear cache
        if self.global_step % self.cfg.get('memory_cleanup_every', 50) == 0:
            torch.cuda.empty_cache()
        
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

    def generate_samples(self, batch, num_samples=5, generate_length=20):
        """Generate and compare longer predictions from teacher and student."""
        self.student_model.eval()
        self.teacher_model.eval()
        
        # Take only a small subset of the batch
        input_ids = batch['input_ids'][:num_samples].to(self.device)
        context = input_ids[:, :-generate_length]
        actual_continuation = input_ids[:, -generate_length:]
        
        samples = []
        # Process one sample at a time to save memory
        for i in range(num_samples):
            with torch.no_grad():
                # Generate one at a time
                student_output = self.student_model.generate(
                    context[i:i+1],
                    max_new_tokens=generate_length,
                    num_beams=4,
                    temperature=0.7,
                )
                
                teacher_output = self.teacher_model.generate(
                    context[i:i+1],
                    max_new_tokens=generate_length,
                    num_beams=4,
                    temperature=0.7,
                )
                
                # Immediately decode and clean to free up GPU memory
                context_text = self.tokenizer.decode(context[i], skip_special_tokens=True)
                actual_text = self.tokenizer.decode(actual_continuation[i], skip_special_tokens=True)
                student_text = self.tokenizer.decode(student_output[0][len(context[i]):], skip_special_tokens=True)
                teacher_text = self.tokenizer.decode(teacher_output[0][len(context[i]):], skip_special_tokens=True)
                
                # Clean texts
                sample = {
                    'context': context_text[-100:],  # Only keep last 100 chars of context
                    'actual_continuation': actual_text[:100],  # Only keep first 100 chars
                    'student_generation': student_text[:100],
                    'teacher_generation': teacher_text[:100]
                }
                samples.append(sample)
                
                # Minimal logging
                if i == 0:  # Only log first sample
                    self.logger.info(f"\nSample generation - Teacher: {teacher_text[:50]}...")
                    self.logger.info(f"Student: {student_text[:50]}...")
        
        # Log to wandb more efficiently
        if self.use_wandb:
            import wandb
            wandb.log({
                "predictions/example": wandb.Table(
                    columns=["Context", "Generated"],
                    data=[[s['context'], s['student_generation']] for s in samples[:2]]  # Only log 2 samples
                )
            }, step=self.global_step)
        
        return samples

    def _compute_text_similarity(self, text1, text2):
        """Compute simple token overlap similarity between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0

    def save_samples(self, samples, epoch, step):
        """Save generated samples and log to wandb if enabled."""
        # Create a formatted table for logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.eval_dir, f"samples_epoch{epoch}_step{step}_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"Sample {i}:\n")
                f.write(f"Context: {sample['context']}\n")
                f.write(f"Actual next token: {sample['actual_continuation']}\n")
                f.write(f"Student prediction: {sample['student_generation']}\n")
                f.write(f"Teacher prediction: {sample['teacher_generation']}\n")
                f.write("-" * 80 + "\n")
        
        if self.use_wandb:
            import wandb
            # Create a wandb Table
            columns = ["Sample", "Context", "Actual", "Student", "Teacher"]
            data = [
                [i+1, 
                 sample['context'], 
                 sample['actual_continuation'],
                 sample['student_generation'],
                 sample['teacher_generation']]
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
        
        for epoch in range(self.epochs_run, self.total_epochs):
            self.student_model.train()
            total_loss = 0
            total_ntp_loss = 0
            total_kd_loss = 0
            logged_steps = 0
            
            progress_bar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Clear cache at the start of accumulation steps
                if batch_idx % self.gradient_accumulation_steps == 0:
                    torch.cuda.empty_cache()
                
                # Forward and backward passes
                loss, ntp_loss, kd_loss = self._loss_step(batch)
                scaled_loss = loss / self.gradient_accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update on accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad_norm is not None:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.student_model.parameters(), self.clip_grad_norm)
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    self.lr_scheduler.step()
                    
                    # Log metrics and generate samples
                    if self.use_wandb and self.global_step % 10 == 0:  # Log every 10 steps
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/ntp_loss": ntp_loss.item(),
                            "train/kd_loss": kd_loss.item(),
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                        }, step=self.global_step)
                    
                    # Generate samples every 500 steps
                    if self.use_wandb and self.global_step % 500 == 0:
                        with torch.no_grad():
                            val_batch = next(iter(self.val_loader))
                            samples = self.generate_samples(val_batch, num_samples=1)
                            wandb.log({
                                "generation/timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "generation/step": self.global_step,
                                "generation/context": samples[0]['context'][-100:],
                                "generation/actual": samples[0]['actual_continuation'][:50],
                                "generation/predicted": samples[0]['student_generation'][:50],
                            }, step=self.global_step)
                            del samples  # Clean up
                            torch.cuda.empty_cache()
                    
                    self.global_step += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'ntp_loss': ntp_loss.item(),
                    'kd_loss': kd_loss.item()
                })
                
                # Track total losses
                total_loss += loss.item()
                total_ntp_loss += ntp_loss.item()
                total_kd_loss += kd_loss.item()
                logged_steps += 1
                
                # Clean up current step
                del loss, ntp_loss, kd_loss, scaled_loss
            
            # End of epoch handling
            progress_bar.close()
            avg_loss = total_loss / logged_steps
            val_loss, val_ppl = self.evaluate(self.val_loader)
            
            # Log epoch metrics
            if self.use_wandb:
                wandb.log({
                    "epoch/train_loss": avg_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_perplexity": val_ppl,
                    "epoch/number": epoch,
                }, step=self.global_step)
                
                # Save learning curves as plot
                fig = plt.figure(figsize=(10, 6))
                plt.plot(self.train_losses, label='Train')
                plt.plot(self.eval_losses, label='Val')
                plt.title('Loss Curves')
                plt.legend()
                wandb.log({"charts/loss_curve": wandb.Image(fig)}, step=self.global_step)
                plt.close(fig)

            self.epochs_run += 1

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
