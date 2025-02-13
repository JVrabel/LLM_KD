import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
                          BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, 
                          get_linear_schedule_with_warmup)
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
from data_setup import setup_dataloaders
import yaml
import argparse
import wandb  # Add wandb import
from model_builder import ModelBuilder

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')



def ddp_setup(rank, world_size): 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    destroy_process_group()

#############################################
# KDRecipe: Knowledge Distillation Recipe
#############################################
class KDRecipe:
    def __init__(self, cfg, rank=None, world_size=None):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if rank is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Initialize model builder with rank
        self.model_builder = ModelBuilder(cfg, rank=rank)
        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None
        
        self.output_dir = cfg['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
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

        # Create a unique run directory based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.eval_dir = os.path.join(self.run_dir, "evaluations")
        os.makedirs(self.eval_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.eval_every = cfg.get('eval_every', 100)
        self.eval_steps = cfg.get('eval_steps', 100)
        self.train_losses = []
        self.eval_losses = []
        self.train_ppls = []
        self.eval_ppls = []
        self.eval_steps_done = 0

        self.save_checkpoint_every = cfg.get('save_checkpoint_every', 5)
        self.keep_n_checkpoints = cfg.get('keep_n_checkpoints', 3)
        self.best_val_loss = float('inf')

        # Initialize wandb only on main process
        self.use_wandb = cfg.get('wandb', {}).get('enabled', False) and (rank is None or rank == 0)
        if self.use_wandb:
            if cfg['resume_from_checkpoint']:
                wandb_run_path = cfg['wandb'].get('resume_id')
                if not wandb_run_path:
                    print("Warning: Resuming training but no wandb run_path provided. Creating new run.")
                
                wandb.init(
                    project=cfg['wandb']['project'],
                    name=cfg['wandb']['name'],
                    id=wandb_run_path,
                    resume="allow",
                    config=cfg,
                    tags=cfg['wandb']['tags'],
                    notes=cfg['wandb']['notes']
                )
            else:
                wandb.init(
                    project=cfg['wandb']['project'],
                    name=cfg['wandb']['name'] or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=cfg,
                    tags=cfg['wandb']['tags'],
                    notes=cfg['wandb']['notes']
                )

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def setup(self):
        # Get models from builder (which now handles DDP wrapping)
        self.tokenizer, self.student_model, self.teacher_model = self.model_builder.setup()
        
        # Get loss functions
        self.ntp_loss_fn, self.kd_loss_fn = self.model_builder.get_loss_functions()
        
        # Setup optimizer after DDP wrapping
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.cfg['learning_rate'])
        
        # Setup data with DistributedSampler if using DDP
        self.train_loader, self.val_loader = self._setup_data()

        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self.steps_per_epoch:
            self.steps_per_epoch = self.max_steps_per_epoch

        self.lr_scheduler = self._setup_lr_scheduler()
        self.scaler = torch.cuda.amp.GradScaler()

    def _setup_data(self):
        if self.rank is not None:
            # Using DDP, create DistributedSampler
            train_loader, val_loader = setup_dataloaders(
                self.cfg, 
                self.tokenizer,
                rank=self.rank,
                world_size=self.world_size
            )
        else:
            # Single GPU setup
            train_loader, val_loader = setup_dataloaders(self.cfg, self.tokenizer)
        return train_loader, val_loader

    def _setup_lr_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_epochs * self.steps_per_epoch),
            num_training_steps=self.total_epochs * self.steps_per_epoch
        )

    def _loss_step(self, batch):
        # Move batch to correct device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()
        
        student_outputs = self.student_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        student_logits = student_outputs.logits[..., :-1, :].contiguous()
        
        # Get labels and shift them right
        labels = batch['labels'][..., 1:].contiguous()
        
        # Calculate NTP loss (next token prediction)
        ntp_loss = self.ntp_loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Calculate KD loss (knowledge distillation)
        kd_loss = self.kd_loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_logits.view(-1, teacher_logits.size(-1)),
            labels.view(-1)
        )

        # Combine losses using kd_ratio
        loss = self.kd_ratio * kd_loss + (1 - self.kd_ratio) * ntp_loss
        return loss, ntp_loss, kd_loss

    def evaluate(self, dataloader, steps=None, desc="Validating"):
        """Run evaluation on the validation set"""
        self.student_model.eval()
        total_loss = 0
        total_ntp_loss = 0
        total_kd_loss = 0
        total_steps = 0
        
        # If steps is None, use the full dataset
        max_steps = steps if steps is not None else len(dataloader)
        
        progress_bar = tqdm(enumerate(dataloader), total=max_steps, desc=desc)
        
        with torch.no_grad():
            for step, batch in progress_bar:
                if step >= max_steps:
                    break
                
                with torch.cuda.amp.autocast():
                    loss, ntp_loss, kd_loss = self._loss_step(batch)
                
                total_loss += loss.item()
                total_ntp_loss += ntp_loss.item()
                total_kd_loss += kd_loss.item()
                total_steps += 1
                
                # Update progress bar without showing intermediate losses
                progress_bar.set_postfix({'steps': f"{step+1}/{max_steps}"})
        
        # Calculate averages
        avg_loss = total_loss / total_steps
        avg_ntp_loss = total_ntp_loss / total_steps
        avg_kd_loss = total_kd_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"Validation results: Loss: {avg_loss:.4f}, NTP Loss: {avg_ntp_loss:.4f}, "
              f"KD Loss: {avg_kd_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        self.student_model.train()
        return {
            'loss': avg_loss,
            'ntp_loss': avg_ntp_loss,
            'kd_loss': avg_kd_loss,
            'perplexity': perplexity
        }

    def generate_samples(self, batch):
        if self.rank is not None and self.rank != 0:
            return None
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Get the base model (unwrap from DDP if needed)
        student_model = self.student_model.module if isinstance(self.student_model, DDP) else self.student_model
        
        with torch.no_grad():
            student_output = student_model.generate(
                input_ids=input_ids[:2],
                attention_mask=attention_mask[:2],
                max_new_tokens=50,  # Generate 50 new tokens instead of using max_length
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            teacher_output = self.teacher_model.generate(
                input_ids=input_ids[:2],
                attention_mask=attention_mask[:2],
                max_new_tokens=50,  # Same for teacher model
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        results = []
        
        with torch.no_grad():
            for i in range(2):  # Generate for 2 examples
                input_ids = batch['input_ids'][i].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'][i].unsqueeze(0).to(self.device)
                
                prompt_length = min(100, input_ids.size(1))
                prompt = input_ids[:, :prompt_length]
                prompt_mask = attention_mask[:, :prompt_length]
                
                # Generate with teacher
                teacher_output = self.teacher_model.generate(
                    input_ids=prompt,
                    attention_mask=prompt_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                
                # Clear cache after teacher generation
                torch.cuda.empty_cache()
                
                # Process outputs
                prompt_text = self.tokenizer.decode(prompt[0], skip_special_tokens=True)
                student_completion = self.tokenizer.decode(student_output[i][prompt_length:], skip_special_tokens=True)
                teacher_completion = self.tokenizer.decode(teacher_output[0][prompt_length:], skip_special_tokens=True)
                ground_truth = self.tokenizer.decode(input_ids[i][prompt_length:], skip_special_tokens=True)
                
                results.append({
                    "prompt": prompt_text,
                    "student_completion": student_completion,
                    "teacher_completion": teacher_completion,
                    "ground_truth": ground_truth
                })
                
                # Clean up tensors
                del student_output, teacher_output, prompt, prompt_mask
                torch.cuda.empty_cache()

        return results

    def train(self):
        for epoch in range(self.epochs_run, self.total_epochs):
            # Set epoch for distributed sampler
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            
            self.student_model.train()
            total_loss = 0
            total_ntp_loss = 0
            total_kd_loss = 0
            logged_steps = 0
            
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                              desc=f"Training", leave=True)
            
            for step, batch in progress_bar:
                if step // self.gradient_accumulation_steps == self.max_steps_per_epoch:
                    break

                with torch.cuda.amp.autocast():
                    loss, ntp_loss, kd_loss = self._loss_step(batch)
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_ntp_loss = ntp_loss / self.gradient_accumulation_steps
                    scaled_kd_loss = kd_loss / self.gradient_accumulation_steps

                self.scaler.scale(scaled_loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.student_model.parameters(), self.clip_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                    # Accumulate the scaled losses
                    total_loss += scaled_loss.item()
                    total_ntp_loss += scaled_ntp_loss.item()
                    total_kd_loss += scaled_kd_loss.item()
                    logged_steps += 1
                    self.global_step += 1

                    # Print training metrics every 10 steps
                    if self.global_step % 10 == 0:
                        progress_bar.set_postfix({
                            'loss': f"{scaled_loss.item():.4f}",
                            'ntp_loss': f"{scaled_ntp_loss.item():.4f}",
                            'kd_loss': f"{scaled_kd_loss.item():.4f}",
                            'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                        })

                    # Validation every eval_every steps
                    if self.global_step % self.eval_every == 0:
                        print(f"\nStep {self.global_step}: Running validation...")
                        val_metrics = self.evaluate(self.val_loader, steps=self.eval_steps)
                        print(f"Validation loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({
                                'val/loss': val_metrics['loss'],
                                'val/ntp_loss': val_metrics['ntp_loss'],
                                'val/kd_loss': val_metrics['kd_loss'],
                                'val/perplexity': val_metrics['perplexity'],
                                'val/step': self.global_step
                            })

                    # Generate and save samples
                    if self.global_step % self.cfg['wandb']['generate_every_n_steps'] == 0:
                        print(f"\nStep {self.global_step}: Generating samples...")
                        samples = self.generate_samples(batch)
                        
                        # Save locally
                        self.save_samples(samples, epoch, self.global_step)
                        
                        # Log to wandb if enabled
                        if self.use_wandb:
                            # Create a wandb.Table for the samples
                            samples_table = wandb.Table(
                                columns=["step", "prompt", "student_completion", "teacher_completion", "ground_truth"],
                                data=[
                                    [self.global_step, s['prompt'], s['student_completion'], 
                                     s['teacher_completion'], s['ground_truth']] for s in samples
                                ]
                            )
                            wandb.log({
                                f"samples/step_{self.global_step}": samples_table,
                            })

                    # Log training metrics to wandb
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': scaled_loss.item(),
                            'train/ntp_loss': scaled_ntp_loss.item(),
                            'train/kd_loss': scaled_kd_loss.item(),
                            'train/learning_rate': self.lr_scheduler.get_last_lr()[0],
                            'train/step': self.global_step,
                        })

            # End of epoch full validation
            print("\nRunning full validation...")
            val_metrics = self.evaluate(self.val_loader)
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    'train/epoch_loss': total_loss / logged_steps,
                    'train/epoch_ntp_loss': total_ntp_loss / logged_steps,
                    'train/epoch_kd_loss': total_kd_loss / logged_steps,
                    'val/loss': val_metrics['loss'],
                    'val/ntp_loss': val_metrics['ntp_loss'],
                    'val/kd_loss': val_metrics['kd_loss'],
                    'val/perplexity': val_metrics['perplexity'],
                    'epoch': epoch
                })
            
            # Save best model based on validation loss
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, total_loss / logged_steps, val_metrics['loss'], is_best=True)
                print(f"New best validation loss: {val_metrics['loss']:.4f}")
            
            # Regular checkpoint saving
            if (epoch + 1) % self.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, total_loss / logged_steps, val_metrics['loss'])
            
            print(f"Epoch {epoch+1} metrics:")
            print(f"Train Loss: {total_loss/logged_steps:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Perplexity: {val_metrics['perplexity']:.4f}")
            
            self.epochs_run += 1

    def _log_metrics(self, loss, ntp_loss, kd_loss, lr):
        print(f"Step {self.global_step}: KD Loss: {kd_loss:.4f}, NTP Loss: {ntp_loss:.4f}, LR: {lr:.6f}")
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

    def save_samples(self, samples, epoch, step):
        samples_file = os.path.join(self.eval_dir, f'samples_epoch_{epoch}_step_{step}.json')
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved samples to {samples_file}")

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        # Only save checkpoint from rank 0 process
        if self.rank is not None and self.rank != 0:
            return

        checkpoint = {
            'epoch': epoch,
            # Use .module to get the underlying model if using DDP
            'student_model_state_dict': self.student_model.module.state_dict() 
                if hasattr(self.student_model, 'module') 
                else self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'train_ppls': self.train_ppls,
            'eval_ppls': self.eval_ppls,
            'wandb_run_id': wandb.run.id if self.use_wandb else None
        }

        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.rank is None or self.rank == 0:  # Only print from main process
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if not is_best:
            # Keep only the N most recent checkpoints
            checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint")])
            for old_checkpoint in checkpoints[:-self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint to CPU first to avoid GPU RAM issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load state dict into model (handle DDP case)
        if hasattr(self.student_model, 'module'):
            self.student_model.module.load_state_dict(checkpoint['student_model_state_dict'])
        else:
            self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.epochs_run = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval_losses = checkpoint.get('eval_losses', [])
        self.train_ppls = checkpoint.get('train_ppls', [])
        self.eval_ppls = checkpoint.get('eval_ppls', [])
        
        # Update wandb run ID in config if available
        if 'wandb_run_id' in checkpoint and checkpoint['wandb_run_id']:
            if 'wandb' not in self.cfg:
                self.cfg['wandb'] = {}
            self.cfg['wandb']['resume_id'] = checkpoint['wandb_run_id']
        
        if self.rank is None or self.rank == 0:  # Only print from main process
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def __del__(self):
        # Cleanup wandb
        if self.use_wandb:
            wandb.finish()

def main(rank=None, world_size=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if rank is not None:
        # Initialize DDP
        ddp_setup(rank, world_size)

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Path to config file')
        parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
        args = parser.parse_args()

        # Print all environment variables at the start
        print("Environment variables:")
        for key, value in os.environ.items():
            if any(x in key.lower() for x in ['cuda', 'nccl', 'rank', 'world', 'master', 'local']):
                print(f"{key}: {value}")

        # Load config
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        # Convert nested yaml config to flat dictionary
        cfg = {
            'model_name': yaml_cfg['model']['name'],
            'model': {
                'student': {
                    'reduce_size': yaml_cfg['model'].get('student', {}).get('reduce_size', True),
                    'size_reduction_factor': yaml_cfg['model'].get('student', {}).get('size_reduction_factor', 2)
                }
            },
            'data_path': yaml_cfg['data']['sources'][0]['path'],
            'output_dir': yaml_cfg['output']['dir'],
            'max_length': yaml_cfg['data']['max_length'],
            'stride': yaml_cfg['data']['stride'],
            'batch_size': yaml_cfg['data']['batch_size'],
            'learning_rate': yaml_cfg['training']['learning_rate'],
            'epochs': yaml_cfg['training']['epochs'],
            'max_steps_per_epoch': yaml_cfg['training']['max_steps_per_epoch'],
            'gradient_accumulation_steps': yaml_cfg['training']['gradient_accumulation_steps'],
            'clip_grad_norm': yaml_cfg['training']['clip_grad_norm'],
            'kd_ratio': yaml_cfg['training']['kd_ratio'],
            'seed': yaml_cfg['training']['seed'],
            'log_every_n_steps': yaml_cfg['training']['log_every_n_steps'],
            'resume_from_checkpoint': args.resume if args.resume else yaml_cfg['checkpointing']['resume'],
            'eval_every': yaml_cfg['training']['eval_every'],
            'eval_steps': yaml_cfg['training']['eval_steps'],
            'save_checkpoint_every': yaml_cfg['checkpointing']['save_every_n_epochs'],
            'keep_n_checkpoints': yaml_cfg['checkpointing']['keep_n_checkpoints'],
            'log_peak_memory_stats': True,
            'training': yaml_cfg['training'],
            'wandb': yaml_cfg['wandb']
        }

        print(f"CUDA available: {torch.cuda.is_available()}")

        recipe = KDRecipe(cfg, rank=rank, world_size=world_size)
        recipe.setup()
        
        if cfg['resume_from_checkpoint']:
            checkpoint_path = cfg['resume_from_checkpoint']
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found at {checkpoint_path}")
                exit(1)
            recipe.load_checkpoint(checkpoint_path)

        recipe.train()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if rank is not None:
            ddp_cleanup()

if __name__ == "__main__":
    # Check if using multiple GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.spawn(
            main,
            args=(n_gpus,),
            nprocs=n_gpus,
            join=True
        )
    else:
        main()  # Run without DDP
