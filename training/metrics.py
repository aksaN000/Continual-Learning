import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class ContinualMetrics:
    """Track metrics for continual learning."""
    
    def __init__(self, num_domains):
        """
        Initialize metrics tracker.
        
        Args:
            num_domains: Number of domains in the continual learning setup
        """
        self.num_domains = num_domains
        
        # Track accuracy for each domain across training stages
        self.domain_accs = [[] for _ in range(num_domains)]
        
        # Track F1 scores (optional)
        self.domain_f1s = [[] for _ in range(num_domains)]
        
        # Current accuracy for each domain
        self.current_accs = [0.0] * num_domains
        
    def update(self, domain_idx, y_true, y_pred):
        """
        Update metrics for a specific domain.
        
        Args:
            domain_idx: Index of the domain
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            acc: Accuracy for this domain
        """
        # Convert tensors to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store metrics for this domain
        self.domain_accs[domain_idx].append(acc)
        self.domain_f1s[domain_idx].append(f1)
        
        # Update current accuracy
        self.current_accs[domain_idx] = acc
        
        return acc
    
    def compute_forgetting(self):
        """
        Compute forgetting measure for each domain.
        Forgetting = max(previous accuracy) - current accuracy
        
        Returns:
            forgetting: List of forgetting measures for each domain
        """
        forgetting = []
        
        for domain_idx in range(self.num_domains):
            accs = self.domain_accs[domain_idx]
            if len(accs) < 2:
                # No forgetting yet for this domain
                forgetting.append(0.0)
            else:
                max_prev_acc = max(accs[:-1])
                current_acc = accs[-1]
                domain_forgetting = max(0, max_prev_acc - current_acc)
                forgetting.append(domain_forgetting)
        
        return forgetting
    
    def compute_avg_accuracy(self):
        """
        Compute average accuracy across all domains seen so far.
        
        Returns:
            avg_acc: Average accuracy
        """
        seen_domains = [i for i, accs in enumerate(self.domain_accs) if accs]
        if not seen_domains:
            return 0.0
        
        current_accs = [self.domain_accs[i][-1] if self.domain_accs[i] else 0.0 
                        for i in seen_domains]
        return np.mean(current_accs)
    
    def compute_backward_transfer(self):
        """
        Compute backward transfer (BWT) metric for each domain.
        BWT = current accuracy - accuracy right after training on that domain
        Positive BWT indicates improved performance after learning more domains.
        Negative BWT indicates forgetting.
        
        Returns:
            bwt: List of backward transfer values for each domain
        """
        bwt = []
        
        for domain_idx in range(self.num_domains):
            accs = self.domain_accs[domain_idx]
            if len(accs) < 2:
                # Not enough data points
                bwt.append(0.0)
            else:
                initial_acc = accs[0]  # Accuracy after training on this domain
                current_acc = accs[-1]  # Current accuracy
                domain_bwt = current_acc - initial_acc
                bwt.append(domain_bwt)
        
        return bwt
    
    def log_metrics(self, current_domain):
        """
        Log current metrics for all domains.
        
        Args:
            current_domain: Index of the domain currently being trained
        """
        print("\n--- Current Performance Metrics ---")
        
        # Print accuracy for each domain seen so far
        print("Accuracy:")
        for i in range(current_domain + 1):
            if self.domain_accs[i]:
                print(f"  Domain {i}: {self.domain_accs[i][-1]:.4f}")
        
        # Print average accuracy
        avg_acc = self.compute_avg_accuracy()
        print(f"Average accuracy: {avg_acc:.4f}")
        
        # Print forgetting measures (only relevant after at least one domain)
        if current_domain > 0:
            forgetting = self.compute_forgetting()
            print("\nForgetting:")
            for i in range(current_domain):
                print(f"  Domain {i}: {forgetting[i]:.4f}")
            
            avg_forgetting = np.mean([forgetting[i] for i in range(current_domain)])
            print(f"Average forgetting: {avg_forgetting:.4f}")