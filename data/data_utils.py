import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class CommandDataset(Dataset):
    """Dataset for text commands with domain labels."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Ensure all texts are strings
        self.texts = [str(text) if text is not None else "" for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Double-check that text is a string
        if not isinstance(text, str):
            text = str(text)
        
        # Handle empty strings
        if not text.strip():
            text = "empty"
        
        # Tokenize the text
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Remove the batch dimension added by the tokenizer
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
            
            return item
        except Exception as e:
            print(f"Error tokenizing text: '{text}', label: {label}")
            print(f"Error details: {e}")
            
            # Return a default empty tokenization
            default_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            default_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            
            return {
                'input_ids': default_input_ids,
                'attention_mask': default_attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }

def get_dataloader(texts, labels, tokenizer, batch_size=16, shuffle=True):
    """Create a DataLoader for the given texts and labels."""
    dataset = CommandDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Tokenize the text
    encoding = self.tokenizer(
        text,
        max_length=self.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Remove the batch dimension added by the tokenizer
    item = {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.long)
    }
    
    return item

def load_domain_data(domain_name, split='train', data_dir='data/processed'):
    """
    Load data for a specific domain and split.
    
    Args:
        domain_name: Name of the domain (folder name)
        split: One of 'train', 'val', or 'test'
        data_dir: Base directory for processed data
        
    Returns:
        texts: List of text commands
        labels: List of corresponding labels
    """
    file_path = os.path.join(data_dir, domain_name, f"{split}.csv")
    
    if os.path.exists(file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert text column to string and handle NaN values
        texts = df['text'].fillna('').astype(str).tolist()
        labels = df['label'].tolist()
        
        return texts, labels
    else:
        print(f"Warning: File not found: {file_path}")
        # Return empty lists if file doesn't exist
        return [], []

def get_num_labels_per_domain(domains, data_dir='data/processed'):
    """
    Get the number of unique labels for each domain.
    
    Args:
        domains: List of domain names
        data_dir: Base directory for processed data
        
    Returns:
        num_labels_per_domain: List of integers
    """
    num_labels_per_domain = []
    
    for domain in domains:
        file_path = os.path.join(data_dir, domain, "train.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            num_labels = len(df['label'].unique())
            num_labels_per_domain.append(num_labels)
        else:
            # If file doesn't exist, assume 3 labels (default)
            print(f"Warning: File not found: {file_path}, assuming 3 labels")
            num_labels_per_domain.append(3)
    
    return num_labels_per_domain