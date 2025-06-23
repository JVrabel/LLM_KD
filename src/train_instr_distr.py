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
import wandb
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
# InstructionKDRecipe: Instruction Tuning with Knowledge Distillation
#############################################
class InstructionKDRecipe:
    def __init__(self, cfg, rank=None, world_size=None):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if rank is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Initialize model builder with instruction teacher model
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
        self.resume_from_checkpoint = cfg['resume_from_checkpoint']  # Required for Phase 2
        self.save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.clip_grad_norm = cfg.get("clip_grad_norm", None)
        self.kd_ratio = cfg.get("kd_ratio", 0.5)
        self.ntp_only = cfg.get('ntp_only', False)

        # Create a unique run directory based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"instruction_run_{timestamp}")
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
            wandb.init(
                project=cfg['wandb']['project'],
                name=f"instruction_{cfg['wandb']['name']}" if cfg['wandb']['name'] else f"instruction_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=cfg,
                tags=cfg['wandb']['tags'] + ['instruction_tuning'],
                notes=f"Instruction tuning phase - {cfg['wandb']['notes']}"
            )

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def setup(self):
        # Get models from builder
        self.tokenizer, self.student_model, self.teacher_model = self.model_builder.setup()
        
        # Load student model from Phase 1 checkpoint
        if self.resume_from_checkpoint:
            print(f"Loading student model from Phase 1 checkpoint: {self.resume_from_checkpoint}")
            self.load_student_checkpoint(self.resume_from_checkpoint)
        else:
            raise ValueError("Instruction tuning requires a checkpoint from Phase 1. Use --resume path/to/checkpoint.pt")
        
        # Get loss functions
        self.ntp_loss_fn, self.kd_loss_fn = self.model_builder.get_loss_functions()
        
        # Setup optimizer after loading checkpoint
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.cfg['learning_rate'])
        
        # Setup data with instruction format
        self.train_loader, self.val_loader = self._setup_data()

        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self.steps_per_epoch:
            self.steps_per_epoch = self.max_steps_per_epoch

        self.lr_scheduler = self._setup_lr_scheduler()
        self.scaler = torch.cuda.amp.GradScaler()

    def load_student_checkpoint(self, checkpoint_path):
        """Load only the student model from Phase 1 checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint to CPU first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load only student model state dict
        if hasattr(self.student_model, 'module'):
            self.student_model.module.load_state_dict(checkpoint['student_model_state_dict'])
        else:
            self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
            
        print(f"Loaded student model from Phase 1 checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

    def _setup_data(self):
        if self.rank is not None:
            train_loader, val_loader = setup_dataloaders(
                self.cfg, 
                self.tokenizer,
                rank=self.rank,
                world_size=self.world_size
            )
        else:
            train_loader, val_loader = setup_dataloaders(self.cfg, self.tokenizer)
        return train_loader, val_loader

    def _setup_lr_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=2000,
            num_training_steps=self.total_epochs * self.steps_per_epoch
        )

    def _loss_step(self, batch):
        # Same as original, but for instruction data
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        if self.ntp_only:
            # NTP-only mode for instruction tuning
            student_outputs = self.student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            student_logits = student_outputs.logits[..., :-1, :].contiguous()
            labels = batch['labels'][..., 1:].contiguous()
            
            ntp_loss = self.ntp_loss_fn(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            
            return ntp_loss, ntp_loss, torch.tensor(0.0, device=ntp_loss.device)
        else:
            # KD mode for instruction tuning
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
            
            labels = batch['labels'][..., 1:].contiguous()
            
            # Calculate NTP loss
            ntp_loss = self.ntp_loss_fn(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            
            # Calculate KD loss
            kd_loss = self.kd_loss_fn(
                student_logits.view(-1, student_logits.size(-1)),
                teacher_logits.view(-1, teacher_logits.size(-1)),
                labels.view(-1)
            )

            # Combine losses
            loss = self.kd_ratio * kd_loss + (1 - self.kd_ratio) * ntp_loss
            return loss, ntp_loss, kd_loss

    # Copy the rest of the methods from train_distr.py (evaluate, generate_samples, train, etc.)
    # ... (same implementation as original)

def main(rank=None, world_size=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if rank is not None:
        ddp_setup(rank, world_size)

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Path to instruction config file')
        parser.add_argument('--resume', type=str, required=True, help='Path to Phase 1 checkpoint to continue from')
        parser.add_argument('--ntp_only', action='store_true', help='Run instruction tuning with only NTP loss')
        args = parser.parse_args()

        # Load instruction config
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        # Convert to flat config (same structure as original)
        cfg = {
            'model_name': yaml_cfg['teacher_model']['name'],  # Instruction teacher model
            'model': {
                'student': {
                    'reduce_size': yaml_cfg['model'].get('student', {}).get('reduce_size', True),
                    'size_reduction_factor': yaml_cfg['model'].get('student', {}).get('size_reduction_factor', 2)
                }
            },
            'data_path': yaml_cfg['data']['instruction_path'],
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
            'resume_from_checkpoint': args.resume,
            'eval_every': yaml_cfg['training']['eval_every'],
            'eval_steps': yaml_cfg['training']['eval_steps'],
            'save_checkpoint_every': yaml_cfg['checkpointing']['save_every_n_epochs'],
            'keep_n_checkpoints': yaml_cfg['checkpointing']['keep_n_checkpoints'],
            'log_peak_memory_stats': True,
            'kd_loss_type': yaml_cfg['training']['kd_loss_type'],
            'kd_temperature': yaml_cfg['training']['kd_temperature'],
            'training': yaml_cfg['training'],
            'wandb': yaml_cfg['wandb'],
            'ntp_only': args.ntp_only
        }

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Phase 2: Instruction Tuning")
        print(f"Teacher model: {cfg['model_name']}")
        print(f"Resuming from: {args.resume}")

        recipe = InstructionKDRecipe(cfg, rank=rank, world_size=world_size)
        recipe.setup()
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
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.spawn(main, args=(n_gpus,), nprocs=n_gpus, join=True)
    else:
        main() 