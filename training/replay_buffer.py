import random
import torch
from collections import deque
import numpy as np
import heapq

class ReplayBuffer:
    """Enhanced memory buffer for experience replay in continual learning."""
    
    def __init__(self, capacity=1000, strategy="balanced"):
        """
        Initialize a replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of examples to store
            strategy: Buffer management strategy: "balanced", "importance", or "diversity"
        """
        self.buffer = deque(maxlen=capacity)
        self.domain_buffers = {}  # Separate buffer per domain
        self.max_capacity = capacity
        self.domain_counts = {}  # Track examples per domain
        self.strategy = strategy
        self.domain_capacity = {}  # Capacity allocated to each domain
        
        # For importance-based sampling
        self.importance_scores = {}
        
        # For diversity sampling
        self.feature_means = None
        self.feature_stds = None
    
    def add_example(self, input_ids, attention_mask, label, domain_idx=None, importance=1.0):
        """
        Add a new example to the buffer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            label: Label tensor
            domain_idx: Domain index (for balanced sampling)
            importance: Importance score (higher = more important to remember)
        """
        example = {
            'input_ids': input_ids.clone().detach(),
            'attention_mask': attention_mask.clone().detach(),
            'label': label.clone().detach(),
            'domain_idx': domain_idx,
            'importance': importance,
            'features': None  # Will be computed later if needed
        }
        
        # If using per-domain buffers, add to the correct one
        if self.strategy == "balanced" and domain_idx is not None:
            if domain_idx not in self.domain_buffers:
                self.domain_buffers[domain_idx] = deque(maxlen=self.max_capacity)
            
            # Add to domain-specific buffer
            self.domain_buffers[domain_idx].append(example)
            
            # Update domain counts
            self.domain_counts[domain_idx] = len(self.domain_buffers[domain_idx])
            
            # Rebalance domain capacities if needed
            self._rebalance_domain_capacities()
        else:
            # Add to main buffer
            self.buffer.append(example)
            
            # Update domain counts if available
            if domain_idx is not None:
                self.domain_counts[domain_idx] = self.domain_counts.get(domain_idx, 0) + 1
    
    def _rebalance_domain_capacities(self):
        """Dynamically adjust capacity per domain based on strategy."""
        num_domains = len(self.domain_buffers)
        if num_domains == 0:
            return
            
        # Default strategy: equal capacity per domain
        base_capacity = self.max_capacity // num_domains
        
        # Allocate capacity per domain
        for domain_idx in self.domain_buffers:
            # Earlier domains get slightly more capacity
            if domain_idx == 0:
                # Give more capacity to the first domain (most prone to forgetting)
                self.domain_capacity[domain_idx] = min(
                    int(base_capacity * 1.5),
                    self.max_capacity // 2  # But not more than half the total
                )
            else:
                # Progressive decrease in capacity for later domains
                decay_factor = 0.9 ** (domain_idx - 1)
                self.domain_capacity[domain_idx] = max(
                    base_capacity // 2,  # Minimum capacity
                    int(base_capacity * decay_factor)
                )
        
        # Adjust buffer sizes if needed
        for domain_idx, buffer in self.domain_buffers.items():
            capacity = self.domain_capacity.get(domain_idx, base_capacity)
            
            # If buffer is larger than capacity, remove least important examples
            if len(buffer) > capacity:
                # Sort by importance and keep the most important ones
                sorted_buffer = sorted(list(buffer), key=lambda x: x.get('importance', 0), reverse=True)
                
                # Create new buffer with adjusted capacity
                new_buffer = deque(maxlen=capacity)
                new_buffer.extend(sorted_buffer[:capacity])
                self.domain_buffers[domain_idx] = new_buffer
                
                # Update domain count
                self.domain_counts[domain_idx] = len(new_buffer)
    
    def _compute_features(self, example):
        """Compute feature representation for diversity measurement."""
        # Use CLS token embedding as feature representation
        input_ids = example['input_ids']
        
        # Just use a simple hash of the input_ids as a feature
        # In a real implementation, you would use the model's embeddings
        feature = torch.mean(input_ids.float(), dim=0)
        
        return feature
    
    def _compute_importance(self, example, correct=None):
        """
        Compute importance score for an example.
        
        Args:
            example: The example to compute importance for
            correct: Whether the prediction was correct (if available)
        
        Returns:
            importance: Importance score (higher = more important)
        """
        # Start with base importance
        importance = 1.0
        
        # If we know prediction correctness, use it
        if correct is not None:
            # Incorrectly classified examples may be more important to remember
            if not correct:
                importance *= 1.5
        
        # Domain-based importance (earlier domains are more important)
        domain_idx = example.get('domain_idx')
        if domain_idx is not None:
            # Exponential decay of importance based on domain index
            importance *= 1.0 / (1.0 + 0.3 * domain_idx)
        
        # TODO: Add more sophisticated importance metrics if needed
        
        return importance
    
    def add_batch(self, batch, device=None, sample_prob=1.0, domain_idx=None, logits=None):
        """
        Add a batch of examples to the buffer with smart selection.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', and 'label'
            device: If specified, moves tensors to this device before storing
            sample_prob: Base probability of adding each example
            domain_idx: Domain index for these examples
            logits: Model predictions (optional, for importance calculation)
        """
        batch_size = batch['input_ids'].size(0)
        
        # If logits are provided, calculate which examples were correctly classified
        correct_predictions = None
        if logits is not None:
            with torch.no_grad():
                # Get predicted classes
                _, predicted = torch.max(logits, 1)
                
                # Compare with ground truth
                correct_predictions = (predicted == batch['label'])
        
        # For each example in the batch
        for i in range(batch_size):
            # Default importance score
            importance = 1.0
            
            # Calculate importance if we have prediction information
            if correct_predictions is not None:
                # Higher importance for examples the model got wrong
                if not correct_predictions[i]:
                    importance = 2.0  # Much higher importance for mistakes
            
            # Adjust sample probability based on importance
            adjusted_prob = min(1.0, sample_prob * importance)
            
            # Only add with certain probability (for better buffer diversity)
            if random.random() < adjusted_prob:
                input_ids = batch['input_ids'][i]
                attention_mask = batch['attention_mask'][i]
                label = batch['label'][i]
                
                # Move to CPU if currently on another device
                if device is not None:
                    input_ids = input_ids.to('cpu')
                    attention_mask = attention_mask.to('cpu')
                    label = label.to('cpu')
                
                self.add_example(input_ids, attention_mask, label, domain_idx, importance)
    
    def _sample_by_importance(self, batch_size):
        """Sample examples based on their importance scores."""
        # If buffer is too small, use all examples
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Calculate sampling probabilities based on importance
        importances = [ex.get('importance', 1.0) for ex in self.buffer]
        total_importance = sum(importances)
        
        if total_importance > 0:
            probs = [imp / total_importance for imp in importances]
            
            # Sample with replacement according to importance
            indices = np.random.choice(
                len(self.buffer), 
                size=batch_size, 
                replace=True, 
                p=probs
            )
            
            return [list(self.buffer)[i] for i in indices]
        else:
            # Fallback to random sampling
            return random.choices(list(self.buffer), k=batch_size)
    
    def _sample_by_diversity(self, batch_size):
        """Sample a diverse set of examples from the buffer."""
        # If buffer is too small, use all examples
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Compute features for all examples if not already done
        for ex in self.buffer:
            if ex['features'] is None:
                ex['features'] = self._compute_features(ex)
        
        # Clustering-based sampling (simplified k-means)
        # Start with a random example
        sampled = [random.choice(list(self.buffer))]
        buffer_list = list(self.buffer)
        
        # Greedily add the most distant examples
        while len(sampled) < batch_size:
            # Find example that maximizes minimum distance to already sampled examples
            best_example = None
            best_min_distance = -float('inf')
            
            for ex in buffer_list:
                if ex in sampled:
                    continue
                    
                # Calculate minimum distance to any sampled example
                min_distance = float('inf')
                for sampled_ex in sampled:
                    # Euclidean distance between feature vectors
                    distance = torch.norm(ex['features'] - sampled_ex['features']).item()
                    min_distance = min(min_distance, distance)
                
                # Keep track of example with maximum minimum distance
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_example = ex
            
            # Add the best example to our sample
            if best_example is not None:
                sampled.append(best_example)
        
        return sampled
    
    def _sample_from_domains(self, batch_size):
        """Sample examples with balanced representation from each domain."""
        # If no domains are tracked, fall back to random sampling
        if not self.domain_counts:
            if len(self.buffer) <= batch_size:
                return list(self.buffer)
            else:
                return random.sample(list(self.buffer), batch_size)
        
        # Get domains and calculate examples per domain
        domains = list(self.domain_counts.keys())
        examples_per_domain = {}
        
        # Allocation strategy: earlier domains get more examples
        total_domains = len(domains)
        remaining = batch_size
        
        for i, domain in enumerate(domains):
            # Earlier domains get more examples (reverse domain order)
            weight = 1.0 + 0.5 * (total_domains - i - 1) / total_domains
            count = max(1, int(batch_size * weight * self.domain_counts[domain] / sum(self.domain_counts.values())))
            
            # Don't allocate more than available or needed
            count = min(count, self.domain_counts[domain], remaining)
            examples_per_domain[domain] = count
            remaining -= count
        
        # If we have remaining examples to allocate, distribute from earliest domains
        for domain in sorted(domains):
            if remaining <= 0:
                break
            
            # Add one more example from this domain if available
            additional = min(remaining, self.domain_counts[domain] - examples_per_domain[domain])
            examples_per_domain[domain] += additional
            remaining -= additional
        
        # Sample from each domain
        sampled = []
        
        # Using domain-specific buffers if available
        if self.domain_buffers:
            for domain, count in examples_per_domain.items():
                if domain in self.domain_buffers and count > 0:
                    domain_buffer = list(self.domain_buffers[domain])
                    
                    # Sample from this domain (with replacement if necessary)
                    if len(domain_buffer) <= count:
                        domain_samples = domain_buffer
                    else:
                        # For more important domains (lower index), prioritize by importance
                        if domain < 2:  # First two domains
                            # Sort by importance and take top examples
                            sorted_buffer = sorted(
                                domain_buffer, 
                                key=lambda x: x.get('importance', 1.0),
                                reverse=True
                            )
                            domain_samples = sorted_buffer[:count]
                        else:
                            # Random sampling for later domains
                            domain_samples = random.sample(domain_buffer, count)
                    
                    sampled.extend(domain_samples)
        else:
            # Using the main buffer
            for domain, count in examples_per_domain.items():
                if count > 0:
                    # Get all examples from this domain
                    domain_examples = [ex for ex in self.buffer if ex['domain_idx'] == domain]
                    
                    # Sample from this domain
                    if len(domain_examples) <= count:
                        sampled.extend(domain_examples)
                    else:
                        sampled.extend(random.sample(domain_examples, count))
        
        # If we couldn't get enough examples with domain balancing, sample randomly
        if len(sampled) < batch_size:
            additional = batch_size - len(sampled)
            remaining_examples = [ex for ex in self.buffer if ex not in sampled]
            
            if remaining_examples:
                sampled.extend(random.sample(remaining_examples, min(additional, len(remaining_examples))))
        
        return sampled
    
    def sample(self, batch_size, device=None):
        """
        Sample a batch of examples from the buffer using the chosen strategy.
        
        Args:
            batch_size: Number of examples to sample
            device: If specified, moves tensors to this device
            
        Returns:
            batch: Dictionary with batched tensors
        """
        if len(self.buffer) == 0 and not self.domain_buffers:
            return None
        
        # Sample examples based on the chosen strategy
        if self.strategy == "balanced":
            sampled = self._sample_from_domains(batch_size)
        elif self.strategy == "importance":
            sampled = self._sample_by_importance(batch_size)
        elif self.strategy == "diversity":
            sampled = self._sample_by_diversity(batch_size)
        else:
            # Default to random sampling
            if len(self.buffer) <= batch_size:
                sampled = list(self.buffer)
            else:
                sampled = random.sample(list(self.buffer), batch_size)
        
        # If we couldn't get enough examples, duplicate some
        if len(sampled) < batch_size:
            sampled = sampled + random.choices(sampled, k=(batch_size - len(sampled)))
        
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
        """Return the total number of examples in all buffers."""
        if self.domain_buffers:
            return sum(len(buffer) for buffer in self.domain_buffers.values())
        else:
            return len(self.buffer)