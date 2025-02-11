import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
import json
import os
from tqdm import tqdm
from distributed_utils import is_distributed

class SlidingWindowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        # Create a cache file name based on the input file, max_length, and stride
        cache_name = os.path.basename(file_path) + f".cache_{max_length}_{stride}.pt"
        self.cache_file = os.path.join(os.path.dirname(file_path), cache_name)
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            self.examples = torch.load(self.cache_file)
        else:
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

    def load_and_preprocess(self, file_path):
        texts = []
        print("Loading and cleaning texts...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Clean text before tokenization
                cleaned_text = self.clean_text(data['text'])
                texts.append(cleaned_text)
        
        examples = []
        print("Tokenizing cleaned texts with sliding window...")
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

def setup_dataloaders(cfg, tokenizer, dist_info=None):
    """Create dataloaders for single or multi-GPU training"""
    dataset = SlidingWindowDataset(
        cfg['data_path'], 
        tokenizer, 
        cfg['max_length'], 
        cfg['stride']
    )
    
    # Split into train (90%) and val (10%)
    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    
    print(f"Dataset split: {train_size} train, {val_size} val examples.")
    
    # Setup samplers for distributed training
    if is_distributed(cfg):
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader