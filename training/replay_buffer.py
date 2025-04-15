import random
import torch
from collections import deque
import numpy as np

class ReplayBuffer:
    """Memory buffer for experience replay in continual learning."""
    
    def __init__(self, capacity=1000):
        """
        Initialize a replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of examples to store
        """
        self.buffer = deque(maxlen=capacity)
        self.domain_counts = {}  # Track examples per domain
    
    def add_example(self, input_ids, attention_mask, label, domain_idx=None):
        """
        Add a new example to the buffer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            label: Label tensor
            domain_idx: Domain index (for balanced sampling)
        """
        example = {
            'input_ids': input_ids.clone().detach(),
            'attention_mask': attention_mask.clone().detach(),
            'label': label.clone().detach(),
            'domain_idx': domain_idx
        }
        
        self.buffer.append(example)
        
        # Update domain counts
        if domain_idx is not None:
            self.domain_counts[domain_idx] = self.domain_counts.get(domain_idx, 0) + 1
    
    def add_batch(self, batch, device=None, sample_prob=1.0, domain_idx=None):
        """
        Add a batch of examples to the buffer.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', and 'label'
            device: If specified, moves tensors to this device before storing
            sample_prob: Probability of adding each example (to control buffer composition)
            domain_idx: Domain index for these examples (for balanced sampling)
        """
        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            # Only add with certain probability (for better buffer diversity)
            if random.random() < sample_prob:
                input_ids = batch['input_ids'][i]
                attention_mask = batch['attention_mask'][i]
                label = batch['label'][i]
                
                # Move to CPU if currently on another device
                if device is not None:
                    input_ids = input_ids.to('cpu')
                    attention_mask = attention_mask.to('cpu')
                    label = label.to('cpu')
                
                self.add_example(input_ids, attention_mask, label, domain_idx)
    
    def sample(self, batch_size, device=None):
        """
        Sample a batch of examples from the buffer, balanced across domains if possible.
        
        Args:
            batch_size: Number of examples to sample
            device: If specified, moves tensors to this device
            
        Returns:
            batch: Dictionary with batched tensors
        """
        if len(self.buffer) == 0:
            return None
            
        if len(self.buffer) < batch_size:
            # If not enough examples, duplicate some
            sampled = random.choices(self.buffer, k=batch_size)
        else:
            # Try to balance across domains if we have domain information
            if self.domain_counts:
                # Determine how many examples to sample from each domain
                domains = list(self.domain_counts.keys())
                examples_per_domain = {}
                
                # Proportional allocation based on buffer composition
                total_examples = sum(self.domain_counts.values())
                remaining = batch_size
                
                for domain in domains[:-1]:  # All but the last domain
                    count = int(batch_size * self.domain_counts[domain] / total_examples)
                    examples_per_domain[domain] = min(count, self.domain_counts[domain])
                    remaining -= examples_per_domain[domain]
                
                # Assign remaining examples to the last domain
                examples_per_domain[domains[-1]] = min(remaining, self.domain_counts[domains[-1]])
                
                # If we still have remaining examples, distribute them randomly
                remaining -= examples_per_domain[domains[-1]]
                if remaining > 0:
                    sampled_domains = random.choices(domains, k=remaining)
                    for domain in sampled_domains:
                        examples_per_domain[domain] += 1
                
                # Sample from each domain
                sampled = []
                for domain, count in examples_per_domain.items():
                    domain_examples = [ex for ex in self.buffer if ex['domain_idx'] == domain]
                    if domain_examples:
                        domain_samples = random.sample(domain_examples, min(count, len(domain_examples)))
                        sampled.extend(domain_samples)
                
                # If we couldn't get enough examples with domain balancing, sample randomly
                if len(sampled) < batch_size:
                    additional = random.sample(self.buffer, batch_size - len(sampled))
                    sampled.extend(additional)
            else:
                # Regular random sampling if no domain information
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