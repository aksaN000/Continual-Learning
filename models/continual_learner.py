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
        
        # Store importance values per domain for more precise regularization
        self.domain_importance = []  # List of importance matrices for each domain
        self.domain_params = []      # List of parameter values for each domain
        
        # Parameter mask for online EWC
        self.parameter_mask = None   # Mask to filter which parameters to regularize
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def get_domain_logits(self, logits, domain_idx):
        """Extract logits for the specified domain."""
        start_idx, end_idx = self.label_ranges[domain_idx]
        return logits[:, start_idx:end_idx]
    
    def compute_ewc_loss(self):
        """
        Compute EWC regularization loss with online EWC improvements.
        
        Uses a weighted sum of importance from all previous domains.
        """
        # If EWC is disabled or not initialized yet, return 0
        if self.ewc_lambda <= 0 or (not self.domain_importance and not self.importance):
            return 0
        
        ewc_loss = 0
        
        # If we're using per-domain importance (enhanced version)
        if self.domain_importance:
            # Separate losses for each previous domain
            domain_losses = []
            
            for domain_idx, (importance, params) in enumerate(zip(self.domain_importance, self.domain_params)):
                domain_loss = 0
                
                # Calculate loss for each parameter
                for name, param in self.model.named_parameters():
                    if name in importance and name in params:
                        # Apply parameter mask if available
                        mask = self.parameter_mask.get(name, None) if self.parameter_mask else None
                        
                        # Calculate squared difference
                        delta = (param - params[name]) ** 2
                        
                        # Apply mask if available
                        if mask is not None:
                            delta = delta * mask
                        
                        # Weight by importance
                        weighted_delta = importance[name] * delta
                        domain_loss += weighted_delta.sum()
                
                # More recent domains might be more important to remember
                # Apply a weighting factor (optional, can be modified)
                recency_weight = 1.0  # Equal weight by default
                domain_losses.append(recency_weight * domain_loss)
            
            # Sum losses from all domains
            if domain_losses:
                ewc_loss = sum(domain_losses)
        
        # If we're using the original (aggregated) importance
        else:
            for name, param in self.model.named_parameters():
                if name in self.importance and name in self.old_params:
                    # Calculate squared difference weighted by importance
                    delta = (param - self.old_params[name]) ** 2
                    weighted_delta = self.importance[name] * delta
                    ewc_loss += weighted_delta.sum()
        
        # Scale by lambda/2 as per the EWC paper
        final_loss = (self.ewc_lambda / 2) * ewc_loss
        return final_loss
    
    def update_ewc_params(self, dataloader, device, domain_idx=None):
        """
        Update EWC parameters based on current domain data.
        Calculate Fisher Information Matrix as the importance.
        
        Args:
            dataloader: DataLoader for the current domain
            device: Computing device
            domain_idx: Current domain index (for per-domain importance)
        """
        # Skip if EWC is disabled
        if self.ewc_lambda <= 0:
            print("EWC is disabled, skipping parameter update.")
            return
            
        print("Updating EWC parameters with enhanced Fisher calculation...")
        
        # ==== Online EWC Implementation ====
        # Store current parameters for this domain
        current_params = {}
        for name, param in self.model.named_parameters():
            current_params[name] = param.clone().detach()
        
        # Calculate parameter mask (focus on important parameters)
        if self.parameter_mask is None:
            self.parameter_mask = {}
            for name, param in self.model.named_parameters():
                # Start with all parameters included
                self.parameter_mask[name] = torch.ones_like(param).to(device)
        
        # Calculate Fisher Information
        self.model.eval()  # Set to evaluation mode
        
        # Initialize importance for this domain
        domain_importance = {}
        for name, param in self.model.named_parameters():
            domain_importance[name] = torch.zeros_like(param).to(device)
        
        # Track number of samples for normalization
        sample_count = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_size = input_ids.size(0)
            sample_count += batch_size
            
            # Process multiple examples in parallel
            # This is more efficient than processing one by one
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Get model's predictions
            probs = F.softmax(outputs, dim=1)
            
            # Get predicted classes
            pred_classes = torch.argmax(probs, dim=1)
            
            # Calculate log probabilities of the predicted classes
            log_probs = F.log_softmax(outputs, dim=1)
            selected_log_probs = log_probs[torch.arange(batch_size), pred_classes]
            
            # Sum the log probabilities
            log_prob_sum = selected_log_probs.sum()
            
            # Backward pass to get gradients
            log_prob_sum.backward()
            
            # Square gradients and accumulate for Fisher Information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Square the gradients (each example contributes)
                    domain_importance[name] += param.grad.pow(2).detach()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches for Fisher calculation")
        
        # Normalize by number of samples
        if sample_count > 0:
            for name in domain_importance:
                domain_importance[name] /= sample_count
        
        # Print statistics about importance values
        print("Fisher Information statistics for current domain:")
        for name in list(domain_importance.keys())[:3]:  # First few layers
            if domain_importance[name].numel() > 0:
                imp = domain_importance[name]
                print(f"  {name}: min={imp.min().item():.6f}, max={imp.max().item():.6f}, mean={imp.mean().item():.6f}")
        
        # Store per-domain importance and parameters
        self.domain_importance.append(domain_importance)
        self.domain_params.append(current_params)
        
        # Update the parameter mask based on accumulated importance
        if domain_idx is not None and domain_idx > 0:
            self._update_parameter_mask()
            
        print(f"EWC lambda set to {self.ewc_lambda}")
        print(f"Number of domains with stored importance: {len(self.domain_importance)}")
    
    def _update_parameter_mask(self, threshold_percentile=90):
        """
        Update parameter mask to focus on the most important parameters.
        
        Args:
            threshold_percentile: Percentile threshold for parameter importance
        """
        # Skip if we don't have multiple domains yet
        if len(self.domain_importance) < 2:
            return
            
        print("Updating parameter importance mask...")
        
        # Aggregate importance across all domains
        aggregated_importance = {}
        for name in self.domain_importance[0]:
            # Initialize with zeros
            aggregated_importance[name] = torch.zeros_like(self.domain_importance[0][name])
            
            # Sum importance across all domains
            for domain_imp in self.domain_importance:
                if name in domain_imp:
                    aggregated_importance[name] += domain_imp[name]
        
        # Calculate threshold for each parameter tensor
        for name, importance in aggregated_importance.items():
            if importance.numel() > 0:
                # Flatten the tensor
                flat_importance = importance.view(-1)
                
                # Calculate threshold value (e.g., 90th percentile)
                k = max(1, int(flat_importance.numel() * (100 - threshold_percentile) / 100))
                threshold_value = torch.kthvalue(flat_importance, k).values.item()
                
                # Update mask based on threshold
                self.parameter_mask[name] = (importance > threshold_value).float()
                
                # Print statistics
                mask_ratio = self.parameter_mask[name].mean().item()
                print(f"  {name}: {mask_ratio:.2%} parameters above threshold")
        
        print(f"Parameter mask updated. Average mask ratio: {sum(mask.mean().item() for mask in self.parameter_mask.values()) / len(self.parameter_mask):.2%}")
        
    def consolidate_ewc_online(self):
        """
        Consolidate EWC importance for online EWC.
        This combines importance from all previous domains for more efficient storage.
        """
        if not self.domain_importance:
            return
            
        print("Consolidating EWC importance matrices...")
        
        # Initialize consolidated importance
        self.importance = {}
        for name in self.domain_importance[0]:
            self.importance[name] = torch.zeros_like(self.domain_importance[0][name])
            
        # Sum importance across all domains (with optional weighting)
        for domain_idx, domain_imp in enumerate(self.domain_importance):
            # Optional: weight by recency (more recent domains are more important)
            weight = 1.0  # Equal weight for all domains
            
            for name, imp in domain_imp.items():
                if name in self.importance:
                    self.importance[name] += weight * imp
        
        # Store the most recent parameters
        self.old_params = self.domain_params[-1] if self.domain_params else {}
        
        # Clear per-domain storage to save memory
        # Note: Comment these lines if you want to keep per-domain information
        # self.domain_importance = []
        # self.domain_params = []
        
        print("EWC importance consolidated.")