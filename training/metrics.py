# training/enhanced_metrics.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

class EnhancedContinualMetrics:
    """
    Enhanced metrics tracking for continual learning.
    Provides comprehensive evaluation metrics for measuring 
    different aspects of continual learning performance.
    """
    
    def __init__(self, num_domains):
        """
        Initialize metrics tracker.
        
        Args:
            num_domains: Number of domains in the continual learning setup
        """
        self.num_domains = num_domains
        
        # Basic performance metrics
        self.domain_accs = [[] for _ in range(num_domains)]
        self.domain_f1s = [[] for _ in range(num_domains)]
        self.domain_precisions = [[] for _ in range(num_domains)]
        self.domain_recalls = [[] for _ in range(num_domains)]
        
        # Confusion matrices
        self.confusion_matrices = [[] for _ in range(num_domains)]
        
        # Resource utilization metrics
        self.training_times = [[] for _ in range(num_domains)]
        self.ewc_computation_times = [[] for _ in range(num_domains)]
        
        # Parameter dynamics
        self.param_importance_matrices = []
        self.param_changes = [[] for _ in range(num_domains)]
        
        # Zero-shot performance (for forward transfer)
        self.zero_shot_accuracies = [0.0] * num_domains
        
        # Learning curves
        self.learning_curves = [[] for _ in range(num_domains)]
        
        # Domain order tracking (for sensitivity analysis)
        self.domain_order = []
        
        # Timestamps for measuring time-based metrics
        self.start_time = time.time()
        self.domain_start_times = []
        
        # Replay buffer statistics
        self.replay_buffer_stats = []
        
    def update(self, domain_idx, y_true, y_pred, model=None, replay_buffer=None, timing_info=None):
        """
        Update metrics for a specific domain.
        
        Args:
            domain_idx: Index of the domain
            y_true: Ground truth labels
            y_pred: Predicted labels
            model: Model being evaluated (for parameter-based metrics)
            replay_buffer: Replay buffer (for utilization metrics)
            timing_info: Dictionary of timing information (optional)
            
        Returns:
            metrics_dict: Dictionary of computed metrics for this update
        """
        # Convert tensors to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Calculate basic metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Store metrics for this domain
        self.domain_accs[domain_idx].append(acc)
        self.domain_f1s[domain_idx].append(f1)
        self.domain_precisions[domain_idx].append(precision)
        self.domain_recalls[domain_idx].append(recall)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[domain_idx].append(cm)
        
        # Track timing information if provided
        if timing_info:
            if 'train_time' in timing_info:
                self.training_times[domain_idx].append(timing_info['train_time'])
            if 'ewc_time' in timing_info:
                self.ewc_computation_times[domain_idx].append(timing_info['ewc_time'])
                
        # Track parameter changes if model is provided
        if model and hasattr(model, 'old_params') and model.old_params:
            param_change_avg = 0.0
            param_count = 0
            
            for name, param in model.named_parameters():
                if name in model.old_params:
                    change = torch.mean(torch.abs(param - model.old_params[name])).item()
                    param_change_avg += change
                    param_count += 1
            
            if param_count > 0:
                param_change_avg /= param_count
                self.param_changes[domain_idx].append(param_change_avg)
        
        # Track replay buffer statistics if provided
        if replay_buffer and hasattr(replay_buffer, '__len__'):
            stats = {
                'buffer_size': len(replay_buffer),
                'domain_distribution': replay_buffer.domain_counts if hasattr(replay_buffer, 'domain_counts') else {}
            }
            self.replay_buffer_stats.append(stats)
        
        # Calculate comprehensive metrics
        metrics_dict = {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'forgetting': self.compute_forgetting()[domain_idx] if domain_idx < len(self.compute_forgetting()) else 0.0,
        }
        
        # Add backward transfer if we have enough data
        if len(self.domain_accs[domain_idx]) >= 2:
            bwt = self.compute_backward_transfer()[domain_idx]
            metrics_dict['backward_transfer'] = bwt
        
        return metrics_dict

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
    
    def compute_maximum_forgetting(self):
        """
        Compute maximum forgetting across all domains.
        
        Returns:
            max_forgetting: Maximum forgetting value
        """
        forgetting = self.compute_forgetting()
        return max(forgetting) if forgetting else 0.0
    
    def compute_forgetting_rate(self):
        """
        Compute forgetting rate for each domain (slope of forgetting).
        
        Returns:
            forgetting_rates: List of forgetting rates for each domain
        """
        forgetting_rates = []
        
        for domain_idx in range(self.num_domains - 1):  # Exclude last domain
            accs = self.domain_accs[domain_idx]
            if len(accs) < 3:  # Need at least 3 points to establish a trend
                forgetting_rates.append(0.0)
                continue
            
            # Use numpy's polyfit to get the slope of the line
            x = np.arange(len(accs))
            slope, _ = np.polyfit(x, accs, 1)
            
            # Negative slope indicates forgetting
            forgetting_rates.append(-slope)
        
        return forgetting_rates
    
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
    
    def compute_avg_f1(self):
        """
        Compute average F1 score across all domains seen so far.
        
        Returns:
            avg_f1: Average F1 score
        """
        seen_domains = [i for i, f1s in enumerate(self.domain_f1s) if f1s]
        if not seen_domains:
            return 0.0
        
        current_f1s = [self.domain_f1s[i][-1] if self.domain_f1s[i] else 0.0 
                      for i in seen_domains]
        return np.mean(current_f1s)
    
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
    
    def compute_avg_backward_transfer(self):
        """
        Compute average backward transfer across all domains.
        
        Returns:
            avg_bwt: Average backward transfer
        """
        bwt_values = self.compute_backward_transfer()
        # Exclude the last domain and any domain with no backward transfer data
        valid_bwt = [bwt for idx, bwt in enumerate(bwt_values) 
                    if idx < self.num_domains - 1 and bwt != 0.0]
        
        return np.mean(valid_bwt) if valid_bwt else 0.0
    
    def compute_forward_transfer(self, zero_shot_accuracies=None):
        """
        Compute forward transfer metric.
        FT = performance on domain immediately after learning previous domains
            - performance on domain without previous knowledge
        
        Args:
            zero_shot_accuracies: List of zero-shot accuracies for each domain (optional)
            
        Returns:
            forward_transfer: List of forward transfer values
        """
        if zero_shot_accuracies:
            self.zero_shot_accuracies = zero_shot_accuracies
            
        forward_transfer = []
        
        # Skip first domain (no previous domains to learn from)
        for domain_idx in range(1, self.num_domains):
            if not self.domain_accs[domain_idx]:
                forward_transfer.append(0.0)
                continue
                
            # Accuracy when first learning this domain
            initial_acc = self.domain_accs[domain_idx][0]
            
            # Zero-shot accuracy (performance without prior knowledge)
            zero_shot_acc = self.zero_shot_accuracies[domain_idx]
            
            # Compute forward transfer
            ft = initial_acc - zero_shot_acc
            forward_transfer.append(ft)
        
        # Add 0 for the first domain (no forward transfer)
        return [0.0] + forward_transfer
    
    def compute_avg_forward_transfer(self):
        """
        Compute average forward transfer across all applicable domains.
        
        Returns:
            avg_ft: Average forward transfer
        """
        ft_values = self.compute_forward_transfer()
        # Skip the first domain
        valid_ft = [ft for ft in ft_values[1:] if ft != 0.0]
        
        return np.mean(valid_ft) if valid_ft else 0.0
    
    def compute_stability(self):
        """
        Compute stability score (1 - average forgetting).
        
        Returns:
            stability: Stability score
        """
        forgetting = self.compute_forgetting()
        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        return 1.0 - avg_forgetting
    
    def compute_plasticity(self):
        """
        Compute plasticity score (ability to learn new domains).
        
        Returns:
            plasticity: Plasticity score
        """
        # Calculate average accuracy on each domain right after learning it
        immediate_accs = []
        for domain_idx in range(self.num_domains):
            if self.domain_accs[domain_idx]:
                immediate_accs.append(self.domain_accs[domain_idx][0])
        
        return np.mean(immediate_accs) if immediate_accs else 0.0
    
    def compute_plasticity_stability_ratio(self):
        """
        Compute ratio between plasticity and stability.
        A balanced model should have a ratio close to 1.
        
        Returns:
            ps_ratio: Plasticity-stability ratio
        """
        plasticity = self.compute_plasticity()
        stability = self.compute_stability()
        
        # Avoid division by zero
        if stability <= 0:
            return float('inf')
            
        return plasticity / stability
    
    def compute_catastrophic_forgetting_events(self, threshold=0.2):
        """
        Count catastrophic forgetting events (sudden large drops in performance).
        
        Args:
            threshold: Drop in accuracy to be considered catastrophic
            
        Returns:
            cf_events: Count of catastrophic forgetting events
        """
        cf_events = 0
        
        for domain_idx in range(self.num_domains):
            accs = self.domain_accs[domain_idx]
            
            if len(accs) < 2:
                continue
                
            # Check for drops in accuracy that exceed the threshold
            for i in range(1, len(accs)):
                if accs[i-1] - accs[i] > threshold:
                    cf_events += 1
        
        return cf_events
    
    def compute_replay_buffer_efficiency(self):
        """
        Compute replay buffer efficiency metrics.
        
        Returns:
            efficiency_metrics: Dictionary of efficiency metrics
        """
        if not self.replay_buffer_stats:
            return {'buffer_utilization': 0.0}
            
        # Calculate average buffer utilization
        utilization = np.mean([stats['buffer_size'] for stats in self.replay_buffer_stats])
        
        # Calculate domain diversity in the buffer
        last_stats = self.replay_buffer_stats[-1]
        domain_counts = last_stats.get('domain_distribution', {})
        domain_diversity = len(domain_counts)
        
        return {
            'buffer_utilization': utilization,
            'domain_diversity': domain_diversity
        }
    
    def compute_training_time_overhead(self):
        """
        Compute the overhead in training time due to continual learning techniques.
        
        Returns:
            overhead: Dictionary of timing overheads
        """
        if not self.training_times or not self.ewc_computation_times:
            return {'ewc_overhead_ratio': 0.0}
            
        # Calculate average training time per domain
        avg_train_time = np.mean([time for domain_times in self.training_times 
                                 for time in domain_times])
        
        # Calculate average EWC computation time
        avg_ewc_time = np.mean([time for domain_times in self.ewc_computation_times 
                               for time in domain_times])
        
        # Calculate overhead ratio
        ewc_overhead_ratio = avg_ewc_time / avg_train_time if avg_train_time > 0 else 0.0
        
        return {
            'avg_train_time': avg_train_time,
            'avg_ewc_time': avg_ewc_time,
            'ewc_overhead_ratio': ewc_overhead_ratio
        }
    
    def compute_parameter_importance_stability(self):
        """
        Compute stability of parameter importance across domains.
        
        Returns:
            importance_stability: Measure of consistency in parameter importance
        """
        if len(self.param_importance_matrices) < 2:
            return 0.0
            
        # Calculate correlations between consecutive importance matrices
        correlations = []
        
        for i in range(1, len(self.param_importance_matrices)):
            prev_imp = self.param_importance_matrices[i-1]
            curr_imp = self.param_importance_matrices[i]
            
            # Calculate correlation for each parameter
            param_correlations = []
            for name in prev_imp:
                if name in curr_imp:
                    # Flatten tensors
                    prev_flat = prev_imp[name].view(-1).cpu().numpy()
                    curr_flat = curr_imp[name].view(-1).cpu().numpy()
                    
                    # Calculate correlation
                    if prev_flat.size > 1 and np.std(prev_flat) > 0 and np.std(curr_flat) > 0:
                        corr, _ = pearsonr(prev_flat, curr_flat)
                        param_correlations.append(corr)
            
            # Average correlation across parameters
            if param_correlations:
                correlations.append(np.mean(param_correlations))
        
        # Average correlation across domains
        return np.mean(correlations) if correlations else 0.0
    
    def log_metrics(self, current_domain):
        """
        Log current metrics for all domains.
        
        Args:
            current_domain: Index of the domain currently being trained
        """
        print("\n--- Performance Metrics after Training on Domain", current_domain, "---")
        
        # Print accuracy for each domain seen so far
        print("Accuracy:")
        for i in range(current_domain + 1):
            if self.domain_accs[i]:
                print(f"  Domain {i}: {self.domain_accs[i][-1]:.4f}")
        
        # Print average accuracy
        avg_acc = self.compute_avg_accuracy()
        print(f"Average accuracy: {avg_acc:.4f}")
        
        # Print F1 scores
        print("\nF1 Score:")
        for i in range(current_domain + 1):
            if self.domain_f1s[i]:
                print(f"  Domain {i}: {self.domain_f1s[i][-1]:.4f}")
        
        # Print forgetting measures (only relevant after at least one domain)
        if current_domain > 0:
            forgetting = self.compute_forgetting()
            print("\nForgetting:")
            for i in range(current_domain):
                print(f"  Domain {i}: {forgetting[i]:.4f}")
            
            avg_forgetting = np.mean([forgetting[i] for i in range(current_domain)])
            print(f"Average forgetting: {avg_forgetting:.4f}")
            
            # Print backward transfer
            bwt = self.compute_backward_transfer()
            print("\nBackward Transfer:")
            for i in range(current_domain):
                print(f"  Domain {i}: {bwt[i]:.4f}")
            
            avg_bwt = np.mean([bwt[i] for i in range(current_domain)])
            print(f"Average backward transfer: {avg_bwt:.4f}")
        
        # Print plasticity-stability metrics
        print("\nPlasticity-Stability Metrics:")
        plasticity = self.compute_plasticity()
        stability = self.compute_stability()
        ps_ratio = self.compute_plasticity_stability_ratio()
        
        print(f"  Plasticity (learning ability): {plasticity:.4f}")
        print(f"  Stability (forgetting resistance): {stability:.4f}")
        print(f"  Plasticity-Stability Ratio: {ps_ratio:.4f}")
        
        # Print catastrophic forgetting events
        cf_events = self.compute_catastrophic_forgetting_events()
        print(f"\nCatastrophic Forgetting Events (>20% drop): {cf_events}")
        
        # Print timing information if available
        if any(self.training_times):
            overhead = self.compute_training_time_overhead()
            print("\nTiming Information:")
            print(f"  Average training time per domain: {overhead.get('avg_train_time', 0):.2f}s")
            if 'ewc_overhead_ratio' in overhead:
                print(f"  EWC overhead ratio: {overhead['ewc_overhead_ratio']:.2f}")
    
    def visualize_metrics(self, domains, output_dir=None):
        """
        Generate visualizations for all metrics.
        
        Args:
            domains: List of domain names
            output_dir: Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up the visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Accuracy Evolution
        self._plot_accuracy_evolution(domains, output_dir)
        
        # 2. Forgetting Measures
        self._plot_forgetting_measures(domains, output_dir)
        
        # 3. Backward Transfer
        self._plot_backward_transfer(domains, output_dir)
        
        # 4. Confusion Matrices (if available)
        if any(self.confusion_matrices):
            self._plot_confusion_matrices(domains, output_dir)
        
        # 5. Parameter Changes (if available)
        if any(self.param_changes):
            self._plot_parameter_changes(domains, output_dir)
        
        # 6. Plasticity-Stability Chart
        self._plot_plasticity_stability(domains, output_dir)
        
        # 7. Overall Performance Summary
        self._plot_performance_summary(domains, output_dir)

    def _plot_accuracy_evolution(self, domains, output_dir):
        """Plot the evolution of accuracy for each domain."""
        plt.figure(figsize=(12, 8))
        
        for i, accs in enumerate(self.domain_accs):
            if accs:
                # Create domain indices starting from when this domain was first trained
                x = list(range(i, i + len(accs)))
                plt.plot(x, accs, marker='o', label=f"Domain {i}: {domains[i]}")
                
                # Annotate final accuracy
                plt.annotate(f"{accs[-1]:.2f}", 
                           (i + len(accs) - 1, accs[-1]),
                           textcoords="offset points", 
                           xytext=(0, 10), 
                           ha='center')
        
        plt.xlabel('Domains Trained')
        plt.ylabel('Accuracy')
        plt.title('Evolution of Domain Accuracy During Sequential Training')
        plt.xticks(range(len(domains)), domains, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'accuracy_evolution.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_forgetting_measures(self, domains, output_dir):
        """Plot forgetting measures for each domain."""
        forgetting = self.compute_forgetting()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(forgetting)), forgetting, color='coral')
        
        # Annotate bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.xlabel('Domain')
        plt.ylabel('Forgetting Measure')
        plt.title('Catastrophic Forgetting by Domain')
        plt.xticks(range(len(domains)), domains, rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'forgetting_measures.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_backward_transfer(self, domains, output_dir):
        """Plot backward transfer for each domain."""
        bwt = self.compute_backward_transfer()
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if x >= 0 else 'red' for x in bwt]
        bars = plt.bar(range(len(bwt)), bwt, color=colors)
        
        # Annotate bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.01 if height >= 0 else height - 0.03,
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.xlabel('Domain')
        plt.ylabel('Backward Transfer')
        plt.title('Backward Transfer by Domain')
        plt.xticks(range(len(domains)), domains, rotation=45)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'backward_transfer.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_confusion_matrices(self, domains, output_dir):
        """Plot confusion matrices for the final state of each domain."""
        for i, cms in enumerate(self.confusion_matrices):
            if not cms:
                continue
                
            # Get the most recent confusion matrix
            cm = cms[-1]
            
            # Create labels based on matrix size
            num_classes = cm.shape[0]
            class_labels = [f"Class {j}" for j in range(num_classes)]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for Domain {i}: {domains[i]}')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_domain_{i}.png'), dpi=300)
                plt.close()
            else:
                plt.show()

    def _plot_parameter_changes(self, domains, output_dir):
        """Plot parameter changes across domains."""
        # Flatten the parameter changes for visualization
        domain_indices = []
        change_values = []
        
        for i, changes in enumerate(self.param_changes):
            if changes:
                for j, change in enumerate(changes):
                    domain_indices.append(i)
                    change_values.append(change)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(domain_indices, change_values, alpha=0.7)
        plt.plot(domain_indices, change_values, 'r--', alpha=0.3)
        
        plt.xlabel('Domain Index')
        plt.ylabel('Average Parameter Change')
        plt.title('Parameter Change Magnitude Across Domains')
        plt.xticks(range(len(domains)), domains, rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'parameter_changes.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_plasticity_stability(self, domains, output_dir):
        """Plot plasticity-stability relationship."""
        plasticity = self.compute_plasticity()
        stability = self.compute_stability()
        
        plt.figure(figsize=(8, 8))
        plt.scatter([stability], [plasticity], s=200, color='blue', alpha=0.7)
        
        plt.xlabel('Stability (1 - Avg. Forgetting)')
        plt.ylabel('Plasticity (Avg. Learning Accuracy)')
        plt.title('Plasticity-Stability Trade-off')
        
        # Add diagonal line representing perfect balance
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Balance')
        
        # Add annotations
        plt.annotate(f"({stability:.2f}, {plasticity:.2f})",
                    (stability, plasticity),
                    textcoords="offset points", 
                    xytext=(10, 10), 
                    ha='left')
        
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'plasticity_stability.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_performance_summary(self, domains, output_dir):
        """Plot overall performance summary."""
        # Collect final metrics
        final_accs = [accs[-1] if accs else 0 for accs in self.domain_accs]
        final_f1s = [f1s[-1] if f1s else 0 for f1s in self.domain_f1s]
        forgetting = self.compute_forgetting()
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Plot metrics
        x = np.arange(len(domains))
        width = 0.25
        
        plt.bar(x - width, final_accs, width, label='Final Accuracy', color='royalblue')
        plt.bar(x, final_f1s, width, label='Final F1 Score', color='green')
        plt.bar(x + width, forgetting, width, label='Forgetting', color='coral')
        
        plt.xlabel('Domain')
        plt.ylabel('Score')
        plt.title('Performance Summary Across Domains')
        plt.xticks(x, domains, rotation=45)
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    def save_metrics(self, output_file):
        """
        Save computed metrics to a JSON file.
        
        Args:
            output_file: Path to save the metrics
        """
        import json
        
        metrics_dict = {
            # Basic performance metrics
            'final_accuracies': [accs[-1] if accs else 0 for accs in self.domain_accs],
            'final_f1_scores': [f1s[-1] if f1s else 0 for f1s in self.domain_f1s],
            'avg_accuracy': self.compute_avg_accuracy(),
            'avg_f1_score': self.compute_avg_f1(),
            
            # Forgetting metrics
            'forgetting': self.compute_forgetting(),
            'max_forgetting': self.compute_maximum_forgetting(),
            'forgetting_rate': self.compute_forgetting_rate(),
            
            # Transfer metrics
            'backward_transfer': self.compute_backward_transfer(),
            'avg_backward_transfer': self.compute_avg_backward_transfer(),
            
            # Plasticity-stability metrics
            'plasticity': self.compute_plasticity(),
            'stability': self.compute_stability(),
            'plasticity_stability_ratio': self.compute_plasticity_stability_ratio(),
            
            # Resource metrics
            'catastrophic_forgetting_events': self.compute_catastrophic_forgetting_events(),
            'replay_efficiency': self.compute_replay_buffer_efficiency(),
            'training_overhead': self.compute_training_time_overhead(),
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        print(f"Metrics saved to {output_file}")