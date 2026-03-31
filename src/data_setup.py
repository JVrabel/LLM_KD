import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

class SlidingWindowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, stride, use_sliding_window=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.use_sliding_window = use_sliding_window
        
        # Create cache name that includes sliding window setting
        cache_suffix = "_sliding" if use_sliding_window else "_no_sliding"
        cache_name = os.path.basename(file_path) + f".cache_{max_length}_{stride}{cache_suffix}_v2.pt"
        self.cache_file = os.path.join(os.path.dirname(file_path), cache_name)
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            self.examples = torch.load(self.cache_file)
        else:
            # Delete old cache file if it exists
            old_cache = os.path.join(os.path.dirname(file_path), 
                                   os.path.basename(file_path) + f".cache_{max_length}_{stride}.pt")
            if os.path.exists(old_cache):
                print(f"Removing old cache file: {old_cache}")
                os.remove(old_cache)
            
            self.examples = self.load_and_preprocess(file_path)
            print(f"Caching dataset to {self.cache_file}")
            torch.save(self.examples, self.cache_file)

    def clean_text(self, text):
        """Clean text before tokenization."""
        # Replace multiple newlines with single space
        text = ' '.join(text.split())
        # Remove special formatting characters while preserving meaningful punctuation
        text = text.replace('_', '')  # Remove underscores used for emphasis
        text = text.replace('(', '').replace(')', '')  # Remove parentheses
        text = text.replace('viz.', 'namely')  # Replace archaic abbreviations
        # Clean up any double spaces
        text = ' '.join(text.split())
        return text.strip()

    def format_instruction_example(self, data):
        """Normalize supported instruction formats into prompt/response text."""
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"

        if isinstance(data, list) and len(data) >= 2:
            prompt = self.clean_text(data[0])
            response = self.clean_text(data[1])
            return {
                'prompt_text': f"{prompt}\n\nAnswer:",
                'full_text': f"{prompt}\n\nAnswer: {response}{eos_token}"
            }

        if isinstance(data, dict):
            if 'instruction' in data and 'output' in data:
                instruction = self.clean_text(data['instruction'])
                output = self.clean_text(data['output'])
                input_text = self.clean_text(data.get('input', '')) if data.get('input') else ""

                prompt_parts = [instruction]
                if input_text:
                    prompt_parts.append(f"Input: {input_text}")

                prompt = "\n\n".join(part for part in prompt_parts if part)
                return {
                    'prompt_text': f"{prompt}\n\nAnswer:",
                    'full_text': f"{prompt}\n\nAnswer: {output}{eos_token}"
                }

            if 'question' in data and 'answer' in data:
                question = self.clean_text(data['question'])
                answer = self.clean_text(data['answer'])
                return {
                    'prompt_text': f"{question}\n\nAnswer:",
                    'full_text': f"{question}\n\nAnswer: {answer}{eos_token}"
                }

            if 'prompt' in data and 'response' in data:
                prompt = self.clean_text(data['prompt'])
                response = self.clean_text(data['response'])
                return {
                    'prompt_text': prompt,
                    'full_text': f"{prompt}{response}{eos_token}"
                }

        return None

    def load_and_preprocess(self, file_path):
        texts = []
        instruction_examples = []
        print("Loading and cleaning texts...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                if self.use_sliding_window:
                    if isinstance(data, dict) and 'text' in data:
                        cleaned_text = self.clean_text(data['text'])
                        texts.append(cleaned_text)
                    elif isinstance(data, list) and len(data) >= 2:
                        formatted_example = self.format_instruction_example(data)
                        texts.append(formatted_example['full_text'])
                    elif isinstance(data, dict) and (
                        ('instruction' in data and 'output' in data) or
                        ('question' in data and 'answer' in data) or
                        ('prompt' in data and 'response' in data)
                    ):
                        formatted_example = self.format_instruction_example(data)
                        texts.append(formatted_example['full_text'])
                    else:
                        print(f"Warning: Unknown pretraining data format: {data}")
                    continue

                formatted_example = self.format_instruction_example(data)
                if formatted_example is not None:
                    instruction_examples.append(formatted_example)
                elif isinstance(data, dict) and 'text' in data:
                    cleaned_text = self.clean_text(data['text'])
                    instruction_examples.append({
                        'prompt_text': "",
                        'full_text': cleaned_text
                    })
                else:
                    print(f"Warning: Unknown instruction data format: {data}")
        
        examples = []
        print("Tokenizing cleaned texts...")
        
        if self.use_sliding_window:
            # Use sliding window (for pretraining)
            print("Using sliding window tokenization...")
            for text in tqdm(texts):
                tokenized = self.tokenizer(
                    text,
                    return_overflowing_tokens=True, 
                    max_length=self.max_length,
                    stride=self.stride,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                for i in range(len(tokenized['input_ids'])):
                    input_ids = tokenized['input_ids'][i]
                    attention_mask = tokenized['attention_mask'][i]
                    examples.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': input_ids.clone()
                    })
        else:
            # No sliding window (for instruction tuning). Mask prompt tokens so
            # the model is only supervised on the assistant response.
            print("Using single-example tokenization...")
            for example in tqdm(instruction_examples):
                prompt_text = example['prompt_text']
                full_text = example['full_text']

                prompt_ids = self.tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                    return_tensors='pt'
                )['input_ids'].squeeze(0)

                tokenized = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                
                input_ids = tokenized['input_ids'].squeeze(0)
                attention_mask = tokenized['attention_mask'].squeeze(0)
                labels = input_ids.clone()

                prompt_length = min(prompt_ids.size(0), labels.size(0))
                labels[:prompt_length] = -100
                labels[attention_mask == 0] = -100

                examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
                
        print(f"Created {len(examples)} examples.")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Use clone().detach() to avoid the warning about tensor construction
        return {
            'input_ids': self.examples[idx]['input_ids'].clone().detach(),
            'attention_mask': self.examples[idx]['attention_mask'].clone().detach(),
            'labels': self.examples[idx]['labels'].clone().detach()
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': labels
    }

def setup_dataloaders(cfg, tokenizer, rank=None, world_size=None):
    # Check if this is instruction tuning (no sliding window for Q&A)
    use_sliding_window = not cfg.get('is_instruction_tuning', False)
    
    dataset = SlidingWindowDataset(
        cfg['data_path'], 
        tokenizer, 
        cfg['max_length'], 
        cfg['stride'],
        use_sliding_window=use_sliding_window
    )
    
    # Split into train (90%) and val (10%)
    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(cfg['seed'])
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg['seed']
    ) if rank is not None else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=cfg['seed']
    ) if rank is not None else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader