import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.load_and_preprocess(file_path)

    def load_and_preprocess(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        # Basic preprocessing: strip whitespace and remove empty lines
        return [text.strip() for text in texts if text.strip()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

def create_dataloaders(file_path: str, tokenizer_name: str = "facebook/llama-7b", max_length: int = 128, batch_size: int = 32, num_workers: int = 4):
    """Creates training DataLoader for text data.

    Args:
        file_path: Path to the text file containing the data.
        tokenizer_name: Name or path of the tokenizer to use.
        max_length: Maximum sequence length for tokenization.
        batch_size: Number of samples per batch in the DataLoader.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        A DataLoader for the text data.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = TextDataset(file_path, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader