import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
                        BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM)
import torch.nn.functional as F

class ModelBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None

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
        
        # Reduce size only if specified in config
        if self.cfg.get('model', {}).get('student', {}).get('reduce_size', False):
            reduction_factor = self.cfg.get('model', {}).get('student', {}).get('size_reduction_factor', 2)
            config.num_hidden_layers = max(1, config.num_hidden_layers // reduction_factor)
            config.intermediate_size = max(1, config.intermediate_size // reduction_factor)
            print(f"Student model reduced to {config.num_hidden_layers} layers and intermediate size {config.intermediate_size}.")
        else:
            print("Student model using same architecture as teacher.")
            
        model = LlamaForCausalLM(config).to(self.device)
        return model

    def _setup_teacher_model(self):
        """Setup teacher model with quantization"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        return model

    def get_loss_functions(self):
        """Return the loss functions needed for training"""
        ntp_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        kd_loss_fn = self.kl_div_loss
        return ntp_loss_fn, kd_loss_fn

    @staticmethod
    def kl_div_loss(student_logits, teacher_logits, labels):
        """Knowledge distillation loss function"""
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='batchmean'
        )
        return loss