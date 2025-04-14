import random
import torch
from collections import deque

class ReplayBuffer:
    """Memory buffer for experience replay in continual learning."""
    
    def __init__(self, capacity=1000):
        """
        Initialize a replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of examples to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add_example(self, input_ids, attention_mask, label):
        """Add a new example to the buffer."""
        self.buffer.append({
            'input_ids': input_ids.clone().detach(),
            'attention_mask': attention_mask.clone().detach(),
            'label': label.clone().detach()
        })
    
    def add_batch(self, batch, device=None):
        """
        Add a batch of examples to the buffer.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', and 'label'
            device: If specified, moves tensors to this device before storing
        """
        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            label = batch['label'][i]
            
            # Move to CPU if currently on another device
            if device is not None:
                input_ids = input_ids.to('cpu')
                attention_mask = attention_mask.to('cpu')
                label = label.to('cpu')
            
            self.add_example(input_ids, attention_mask, label)
    
    def sample(self, batch_size, device=None):
        """
        Sample a batch of examples from the buffer.
        
        Args:
            batch_size: Number of examples to sample
            device: If specified, moves tensors to this device
            
        Returns:
            batch: Dictionary with batched tensors
        """
        if len(self.buffer) < batch_size:
            # If not enough examples, duplicate some
            sampled = random.choices(self.buffer, k=batch_size)
        else:
            sampled = random.sample(self.buffer, batch_size)
        
        # Combine into batches
        batch = {
            'input_ids': torch.stack([ex['input_ids'] for ex in sampled]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in sampled]),
            'label': torch.stack([ex['label'] for ex in sampled])
        }
        
        # Move to specified device if needed
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        
        return batch
    
    def __len__(self):
        return len(self.buffer)