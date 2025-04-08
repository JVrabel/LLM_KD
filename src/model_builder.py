import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
                        BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM)
import torch.nn.functional as F
import torch.nn as nn

class ModelBuilder:
    def __init__(self, cfg, rank=None):
        self.cfg = cfg
        self.rank = rank
        # If rank is provided, use specific GPU, otherwise use cuda
        self.device = f'cuda:{rank}' if rank is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None

        # --- Add KD config reading ---
        self.kd_loss_type = self.cfg.get('kd_loss_type', 'mse') # Default to 'mse'
        self.kd_temperature = self.cfg.get('kd_temperature', 2.0) # Default temperature for KL Div
        # --- End of added KD config reading ---

    def setup(self):
        """Setup all model components"""
        self.tokenizer = self._setup_tokenizer()
        self.student_model = self._setup_student_model()
        self.teacher_model = self._setup_teacher_model()
        return self.tokenizer, self.student_model, self.teacher_model

    def _setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _setup_student_model(self):
        """Create student model with configurable size"""
        config = LlamaConfig.from_pretrained(self.cfg['model_name'])
        
        if self.cfg.get('model', {}).get('student', {}).get('reduce_size', False):
            reduction_factor = self.cfg.get('model', {}).get('student', {}).get('size_reduction_factor', 2)
            config.num_hidden_layers = max(1, config.num_hidden_layers // reduction_factor)
            config.intermediate_size = max(1, config.intermediate_size // reduction_factor)
            print(f"Student model reduced to {config.num_hidden_layers} layers and intermediate size {config.intermediate_size}.")
        else:
            print("Student model using same architecture as teacher.")
            
        model = LlamaForCausalLM(config).to(self.device)
        
        # Wrap model in DDP if rank is provided
        if self.rank is not None:
            model = DDP(model, device_ids=[self.rank])
            
        return model

    def _setup_teacher_model(self):
        """Setup teacher model with quantization"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Place teacher model on the same device as the student model
        device = f'cuda:{self.rank}' if self.rank is not None else 'cuda'
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": device}  # Explicitly assign to specific GPU
        )
        model.eval()
        return model

    def get_loss_functions(self):
        """
        Returns the loss functions for training.
        NTP Loss: CrossEntropyLoss
        KD Loss: Configurable via cfg['kd_loss_type'] ('mse' or 'kl_div').
        """
        # Standard Next Token Prediction loss
        # Use pad_token_id from tokenizer if available and different from -100
        # For now, assuming default ignore_index
        ntp_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # --- Define KD Loss Options ---

        # Option 1: MSE Logit Matching
        mse_loss = nn.MSELoss()
        def mse_kd_loss(student_logits, teacher_logits, labels=None):
            # labels are ignored by MSELoss but included for consistent signature
            return mse_loss(student_logits, teacher_logits)

        # Option 2: KL Divergence Loss
        def kl_div_kd_loss(student_logits, teacher_logits, labels=None):
            # Apply temperature scaling and softmax
            # Ensure logits are float for softmax/log_softmax if using AMP/lower precision
            soft_teacher_logits = F.softmax(teacher_logits.float() / self.kd_temperature, dim=-1)
            log_soft_student_logits = F.log_softmax(student_logits.float() / self.kd_temperature, dim=-1)

            # Calculate KL divergence loss
            # Note: F.kl_div expects log-probabilities as input, probabilities as target.
            # The loss is scaled by T^2 according to the original Hinton paper.
            kl_loss = F.kl_div(log_soft_student_logits, soft_teacher_logits, reduction='batchmean') * (self.kd_temperature ** 2)
            return kl_loss

        # --- Select KD Loss based on config ---
        if self.kd_loss_type.lower() == 'mse':
            kd_loss_fn = mse_kd_loss
            print(f"Rank {self.rank}: Using MSELoss for Knowledge Distillation (Logit Matching).") # Added rank for clarity in DDP
        elif self.kd_loss_type.lower() == 'kl_div':
            kd_loss_fn = kl_div_kd_loss
            print(f"Rank {self.rank}: Using KL Divergence Loss for Knowledge Distillation (Temperature: {self.kd_temperature}).") # Added rank
        else:
            # Default or fallback if config value is invalid
            print(f"Rank {self.rank}: Warning: Unknown kd_loss_type '{self.kd_loss_type}'. Defaulting to MSELoss.") # Added rank
            kd_loss_fn = mse_kd_loss
            self.kd_loss_type = 'mse' # Update the internal state

        return ntp_loss_fn, kd_loss_fn