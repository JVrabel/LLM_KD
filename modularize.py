import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.load_and_preprocess(file_path)

    def load_and_preprocess(self, file_path):
        texts = []
        total_tokens = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if text:
                    texts.append(text)
                    total_tokens += len(self.tokenizer.encode(text))
        
        print(f"Loaded {len(texts)} text samples.")
        print(f"Total tokens: {total_tokens}")
        print(f"Estimated pretraining size: {total_tokens / 1e6:.2f}M tokens")
        
        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

def create_dataloader(file_path, tokenizer, max_length=128, batch_size=32, num_workers=4):
    dataset = JSONLDataset(file_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "meta-llama/Llama-2-7b-hf"  # or the path to your local model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    # Create dataloader
    file_path = "data/output/pretraining.jsonl"  # Update this to your JSONL file path
    dataloader = create_dataloader(file_path, tokenizer)

    # Extract logits for each batch
    all_logits = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Extract logits
            logits = outputs.logits
            
            # Move logits to CPU and convert to float32 for storage
            all_logits.append(logits.cpu().float())

            print(f"Processed batch {i+1}/{len(dataloader)}. Logits shape: {logits.shape}")

    # Concatenate all logits
    all_logits = torch.cat(all_logits, dim=0)
    print(f"Total logits shape: {all_logits.shape}")

    # Here you can save the logits or perform further processing
    # For example, to save the logits:
    # torch.save(all_logits, 'llama_logits.pt')

if __name__ == "__main__":
    main()