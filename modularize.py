import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import json
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import warnings
import matplotlib.pyplot as plt
import datetime

class SlidingWindowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = self.load_and_preprocess(file_path)

    def load_and_preprocess(self, file_path):
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])

        examples = []
        for text in texts:
            tokenized = self.tokenizer(text, return_overflowing_tokens=True, 
                                       max_length=self.max_length, stride=self.stride,
                                       truncation=True, padding='max_length', return_tensors='pt')
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

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def create_dataloaders(file_path, tokenizer, max_length=512, stride=256, batch_size=2, num_workers=2):
    dataset = SlidingWindowDataset(file_path, tokenizer, max_length, stride)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def create_small_model(vocab_size):
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config)
    return model

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k, dim=-1)
    return torch.clamp(logits, min=v[..., [-1]])

def kl_div_loss(student_logits, teacher_logits, temperature=1.0, k=200):
    student_logits = top_k_logits(student_logits, k)
    teacher_logits = top_k_logits(teacher_logits, k)
    
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    
    return torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)

@torch.no_grad()
def get_teacher_outputs(teacher_model, batch):
    return teacher_model(**batch)

def evaluate_and_store(student_model, teacher_model, batch, tokenizer, eval_dir, epoch, step):
    student_model.eval()
    teacher_model.eval()
    
    results = []
    
    with torch.no_grad():
        for i in range(2):  # Do 2 examples
            input_ids = batch['input_ids'][i].unsqueeze(0)
            attention_mask = batch['attention_mask'][i].unsqueeze(0)
            
            # Get the prompt (first 100 tokens)
            prompt_length = min(100, input_ids.size(1))
            prompt = input_ids[:, :prompt_length]
            prompt_mask = attention_mask[:, :prompt_length]
            
            # Generate completions
            student_output = student_model.generate(
                input_ids=prompt,
                attention_mask=prompt_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            teacher_output = teacher_model.generate(
                input_ids=prompt,
                attention_mask=prompt_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            
            # Decode texts
            prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)
            student_completion = tokenizer.decode(student_output[0][prompt_length:], skip_special_tokens=True)
            teacher_completion = tokenizer.decode(teacher_output[0][prompt_length:], skip_special_tokens=True)
            ground_truth = tokenizer.decode(input_ids[0][prompt_length:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt_text,
                "student_completion": student_completion,
                "teacher_completion": teacher_completion,
                "ground_truth": ground_truth
            })
    
    # Save results to JSON file
    epoch_dir = os.path.join(eval_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    with open(os.path.join(epoch_dir, f'eval_step_{step}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    student_model.train()

def train_epoch(student_model, teacher_model, dataloader, optimizer, device, scaler, tokenizer, eval_dir, epoch, alpha=0.5, accumulation_steps=4):
    student_model.train()
    teacher_model.eval()
    total_loss = 0
    total_ntp_loss = 0
    total_kd_loss = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            teacher_outputs = get_teacher_outputs(teacher_model, batch)
            teacher_logits = teacher_outputs.logits
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits
            
            ntp_loss = torch.nn.functional.cross_entropy(student_logits.view(-1, student_logits.size(-1)), batch['input_ids'].view(-1))
            kd_loss = kl_div_loss(student_logits, teacher_logits)
            
            loss = (alpha * ntp_loss + (1 - alpha) * kd_loss) / accumulation_steps

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_ntp_loss += ntp_loss.item()
        total_kd_loss += kd_loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"\nStep {i+1}/{num_batches}")
            print(f"Total Loss: {loss.item() * accumulation_steps:.4f}")
            print(f"NTP Loss: {ntp_loss.item():.4f}")
            print(f"KD Loss: {kd_loss.item():.4f}")
            evaluate_and_store(student_model, teacher_model, batch, tokenizer, eval_dir, epoch, i+1)
        
        del teacher_outputs, teacher_logits, student_outputs, student_logits
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    avg_ntp_loss = total_ntp_loss / num_batches
    avg_kd_loss = total_kd_loss / num_batches
    
    return avg_loss, avg_ntp_loss, avg_kd_loss

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch['input_ids'].view(-1))
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "meta-llama/Llama-3.1-8B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()
    teacher_model.resize_token_embeddings(len(tokenizer))

    student_model = create_small_model(len(tokenizer))
    student_model.to(device)

    file_path = "data/pretraining.jsonl"
    train_loader, val_loader, test_loader = create_dataloaders(file_path, tokenizer, max_length=512, batch_size=2)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    num_epochs = 100

    # Create a unique run directory based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    eval_dir = os.path.join(run_dir, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)

    save_dir = os.path.join(run_dir, "models/student")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        'train_losses': [],
        'train_ntp_losses': [],
        'train_kd_losses': [],
        'val_losses': [],
        'test_loss': None
    }

    for epoch in range(num_epochs):
        train_loss, train_ntp_loss, train_kd_loss = train_epoch(student_model, teacher_model, train_loader, optimizer, device, scaler, tokenizer, eval_dir, epoch)
        val_loss = validate(student_model, val_loader, device)
        
        results['train_losses'].append(train_loss)
        results['train_ntp_losses'].append(train_ntp_loss)
        results['train_kd_losses'].append(train_kd_loss)
        results['val_losses'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train NTP Loss: {train_ntp_loss:.4f}")
        print(f"Train KD Loss: {train_kd_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    test_loss = validate(student_model, test_loader, device)
    results['test_loss'] = test_loss
    print(f"Test Loss: {test_loss:.4f}")

    student_model.save_pretrained(save_dir)

    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f)

    plt.figure(figsize=(10, 5))
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['train_ntp_losses'], label='Train NTP Loss')
    plt.plot(results['train_kd_losses'], label='Train KD Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

if __name__ == "__main__":
    main()