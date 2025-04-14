import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.base_model import TextCommandClassifier

class ContinualTextCommandLearner(nn.Module):
    """Continual learning model with replay and regularization."""
    
    def __init__(self, num_labels_per_domain, model_name='bert-base-uncased'):
        super(ContinualTextCommandLearner, self).__init__()
        
        # Total number of labels across all domains
        self.total_labels = sum(num_labels_per_domain)
        
        # Base model
        self.model = TextCommandClassifier(self.total_labels, model_name)
        
        # Store number of labels per domain for later use
        self.num_labels_per_domain = num_labels_per_domain
        self.num_domains = len(num_labels_per_domain)
        
        # Track label ranges for each domain
        self.label_ranges = []
        start_idx = 0
        for num_labels in num_labels_per_domain:
            end_idx = start_idx + num_labels
            self.label_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Parameters for EWC (Elastic Weight Consolidation)
        self.importance = {}  # Parameter importance
        self.old_params = {}  # Previous parameter values
        self.ewc_lambda = 0.0  # Regularization strength (0 = disabled by default)
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def get_domain_logits(self, logits, domain_idx):
        """Extract logits for the specified domain."""
        start_idx, end_idx = self.label_ranges[domain_idx]
        return logits[:, start_idx:end_idx]
    
    def compute_ewc_loss(self):
        """
        Compute EWC regularization loss.
        Loss = λ/2 * sum_i F_i * (θ_i - θ*_i)^2
        where F_i is the importance of parameter i,
        θ_i is the current value, and θ*_i is the old value.
        """
        if self.ewc_lambda == 0 or not self.importance or not self.old_params:
            return 0
        
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.importance and name in self.old_params:
                # Calculate squared difference weighted by importance
                delta = (param - self.old_params[name]) ** 2
                weighted_delta = self.importance[name] * delta
                ewc_loss += weighted_delta.sum()
        
        # Scale by lambda/2 as per the EWC paper
        final_loss = (self.ewc_lambda / 2) * ewc_loss
        return final_loss
    
    def update_ewc_params(self, dataloader, device):
        """
        Update EWC parameters based on current domain data.
        Calculate Fisher Information Matrix as the importance.
        """
        print("Updating EWC parameters with proper Fisher calculation...")
        
        # Store current parameters before moving to next domain
        self.old_params = {}
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.clone().detach()
        
        # Initialize importance (Fisher Information) to zeros
        self.importance = {}
        for name, param in self.model.named_parameters():
            self.importance[name] = torch.zeros_like(param).to(device)
        
        # Calculate Fisher Information
        self.model.eval()  # Set to evaluation mode
        
        # Track number of samples for normalization
        sample_count = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_size = input_ids.size(0)
            sample_count += batch_size
            
            # Process each example individually
            for i in range(batch_size):
                # Zero out gradients
                self.model.zero_grad()
                
                # Forward pass for single example
                single_input = input_ids[i:i+1]  # Keep batch dimension
                single_mask = attention_mask[i:i+1]
                
                outputs = self.model(single_input, single_mask)
                
                # Get model's prediction (important: use predicted class, not ground truth)
                probs = F.softmax(outputs, dim=1)
                
                # Sample from the output distribution (or just take the max)
                # For EWC, we should use the model's own prediction
                pred_class = torch.argmax(probs, dim=1)
                
                # Calculate log probability of the predicted class
                log_prob = F.log_softmax(outputs, dim=1)[0, pred_class]
                
                # Backward pass to get gradients
                log_prob.backward()
                
                # Square gradients and accumulate for Fisher Information
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.importance[name] += param.grad.pow(2).detach()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches for Fisher calculation")
        
        # Normalize by number of samples
        if sample_count > 0:
            for name in self.importance:
                self.importance[name] /= sample_count
        
        # Set EWC lambda to 100 (stronger regularization)
        self.ewc_lambda = 100.0
        
        # Print statistics about importance values
        print("Fisher Information statistics:")
        for name in list(self.importance.keys())[:3]:  # First few layers
            if self.importance[name].numel() > 0:
                imp = self.importance[name]
                print(f"  {name}: min={imp.min().item():.6f}, max={imp.max().item():.6f}, mean={imp.mean().item():.6f}")
        
        print(f"EWC lambda set to {self.ewc_lambda}")