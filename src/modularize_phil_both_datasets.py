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

#############################################
# SlidingWindowDataset with caching
#############################################
class SlidingWindowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        # Create a cache file name based on the input file, max_length, and stride.
        cache_name = os.path.basename(file_path) + f".cache_{max_length}_{stride}.pt"
        self.cache_file = os.path.join(os.path.dirname(file_path), cache_name)
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            self.examples = torch.load(self.cache_file)
        else:
            self.examples = self.load_and_preprocess(file_path)
            print(f"Caching dataset to {self.cache_file}")
            torch.save(self.examples, self.cache_file)

    def load_and_preprocess(self, file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
        examples = []
        for text in texts:
            tokenized = self.tokenizer(
                text,
                return_overflowing_tokens=True, 
                max_length=self.max_length,
                stride=self.stride,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            # For each sliding-window chunk from this text…
            for i in range(len(tokenized['input_ids'])):
                examples.append({
                    'input_ids': tokenized['input_ids'][i],
                    'attention_mask': tokenized['attention_mask'][i]
                })
        print(f"Created {len(examples)} examples with sliding window.")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

#############################################
# Collate function for DataLoader
#############################################
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

#############################################
# KDRecipeSingleDevice: Knowledge Distillation Recipe
#############################################
class KDRecipeSingleDevice:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use bfloat16 if supported, else float32 (or float16 if you prefer)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        self.output_dir = cfg['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if needed
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

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_name'])
        # For causal LM, use EOS as pad token.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up the student model with a smaller architecture.
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
        # Create a smaller student model by reducing some dimensions.
        config = LlamaConfig.from_pretrained(self.cfg['model_name'])
        # Example: reduce number of hidden layers and intermediate size
        config.num_hidden_layers = max(1, config.num_hidden_layers // 2)
        config.intermediate_size = max(1, config.intermediate_size // 2)
        model = LlamaForCausalLM(config)
        print(f"Student model reduced to {config.num_hidden_layers} layers and intermediate size {config.intermediate_size}.")
        return model.to(self.device)

    def _setup_teacher_model(self):
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

    def _setup_data(self):
        # Use the cached sliding window dataset.
        dataset = SlidingWindowDataset(self.cfg['data_path'], self.tokenizer, self.cfg['max_length'], self.cfg['stride'])
        # Split into train (70%), val (10%), test (20%)
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test examples.")
        
        train_loader = DataLoader(train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg['batch_size'], shuffle=False, collate_fn=collate_fn)
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

        # Shift student logits and labels for next-token prediction
        shifted_logits = student_logits[..., :-1, :].contiguous()
        shifted_labels = input_ids[..., 1:].contiguous()

        ntp_loss = self.ntp_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        kd_loss = self.kd_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                  teacher_logits[..., :-1, :].contiguous().view(-1, teacher_logits.size(-1)),
                                  shifted_labels.view(-1))
        # You can combine losses using kd_ratio; here we use the KD loss only.
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
        return avg_loss, ppl

    def generate_samples(self, batch):
        self.student_model.eval()
        self.teacher_model.eval()
        results = []
        
        with torch.no_grad():
            for i in range(2):  # Generate for 2 examples
                input_ids = batch['input_ids'][i].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'][i].unsqueeze(0).to(self.device)
                
                prompt_length = min(100, input_ids.size(1))
                prompt = input_ids[:, :prompt_length]
                prompt_mask = attention_mask[:, :prompt_length]
                
                student_output = self.student_model.generate(
                    input_ids=prompt,
                    attention_mask=prompt_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                teacher_output = self.teacher_model.generate(
                    input_ids=prompt,
                    attention_mask=prompt_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                
                prompt_text = self.tokenizer.decode(prompt[0], skip_special_tokens=True)
                student_completion = self.tokenizer.decode(student_output[0][prompt_length:], skip_special_tokens=True)
                teacher_completion = self.tokenizer.decode(teacher_output[0][prompt_length:], skip_special_tokens=True)
                ground_truth = self.tokenizer.decode(input_ids[0][prompt_length:], skip_special_tokens=True)
                
                results.append({
                    "prompt": prompt_text,
                    "student_completion": student_completion,
                    "teacher_completion": teacher_completion,
                    "ground_truth": ground_truth
                })
        return results

    def train(self):
        for epoch in range(self.epochs_run, self.total_epochs):
            self.student_model.train()
            total_loss = 0
            total_ntp_loss = 0
            total_kd_loss = 0
            logged_steps = 0
            
            pbar = tqdm(total=self.steps_per_epoch, desc=f"Epoch {epoch+1}/{self.total_epochs}")
            
            for step, batch in enumerate(self.train_loader):
                # Stop if we reach the maximum steps per epoch
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
                        self._log_metrics(loss.item() * self.gradient_accumulation_steps, ntp_loss.item(), kd_loss.item(), self.optimizer.param_groups[0]['lr'])

                    if self.global_step % self.eval_every == 0:
                        eval_loss, eval_ppl = self.evaluate(self.val_loader, steps=self.eval_steps)
                        train_loss = total_loss / logged_steps
                        train_ppl = torch.exp(torch.tensor(train_loss)).item()
                        
                        self.train_losses.append(train_loss)
                        self.eval_losses.append(eval_loss)
                        self.train_ppls.append(train_ppl)
                        self.eval_ppls.append(eval_ppl)
                        self.eval_steps_done += 1
                        
                        print(f"Step {self.global_step}: Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}, Eval Loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.4f}")
                        
                        self.plot_metrics()
                        samples = self.generate_samples(batch)
                        self.save_samples(samples, epoch, self.global_step)

                    self.global_step += 1
            pbar.close()
            
            avg_loss = total_loss / self.steps_per_epoch
            avg_ntp_loss = total_ntp_loss / self.steps_per_epoch
            avg_kd_loss = total_kd_loss / self.steps_per_epoch
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg NTP Loss: {avg_ntp_loss:.4f}, Avg KD Loss: {avg_kd_loss:.4f}")
            self.epochs_run += 1

            val_loss, val_ppl = self.evaluate(self.val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Validation PPL: {val_ppl:.4f}")

            if (epoch + 1) % self.save_checkpoint_every == 0 or val_loss < self.best_val_loss:
                self.save_checkpoint(epoch, avg_loss, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, avg_loss, val_loss, is_best=True)

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

    def save_samples(self, samples, epoch, step):
        samples_file = os.path.join(self.eval_dir, f'samples_epoch_{epoch}_step_{step}.json')
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved samples to {samples_file}")

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
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
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        if not is_best:
            # Keep only the N most recent checkpoints.
            checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint")])
            for old_checkpoint in checkpoints[:-self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

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

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg = {
        'model_name': "meta-llama/Llama-3.2-1B",
        'data_path': "data/test/output/pretraining.jsonl",
        'output_dir': "runs/kd_experiment",
        'max_length': 512,
        'stride': 380,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'epochs': 100,
        'max_steps_per_epoch': None,
        'gradient_accumulation_steps': 4,
        'clip_grad_norm': 1.0,
        'kd_ratio': 0.5,
        'seed': 42,
        'log_every_n_steps': 100,
        'resume_from_checkpoint': False,
        'eval_every': 200,
        'eval_steps': 200,
        'save_checkpoint_every': 5,
        'keep_n_checkpoints': 3,
    }

    recipe = KDRecipeSingleDevice(cfg)
    recipe.setup()

    if cfg['resume_from_checkpoint']:
        checkpoint_path = os.path.join(recipe.checkpoint_dir, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(recipe.checkpoint_dir, "checkpoint_epoch_latest.pt")
        recipe.load_checkpoint(checkpoint_path)

    recipe.train()

if __name__ == "__main__":
    main()
