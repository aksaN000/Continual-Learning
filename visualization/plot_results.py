import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

def set_plot_style():
    """Set consistent plot styling."""
    # Use Seaborn's default theme
    sns.set_theme(style="whitegrid")
    # Adjust matplotlib parameters for consistent look
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.figsize': (14, 8),
    })

def plot_accuracy_over_time(accuracies, domains, title=None, save_path=None):
    """
    Plot accuracy for each domain as we train on more domains.
    
    Args:
        accuracies: List of lists, where accuracies[i][j] is the accuracy
                   of domain i after training on domain j
        domains: List of domain names
        title: Plot title (optional)
        save_path: Path to save the plot, if None, the plot is displayed
    """
    # Set plot style
    set_plot_style()
    plt.figure()
    
    # Color palette
    colors = sns.color_palette("Set2", len(domains))
    
    # Plot accuracy for each domain
    for i, domain_accs in enumerate(accuracies):
        if not domain_accs:  # Skip empty accuracy lists
            continue
            
        # Create x values that match the length of domain_accs
        x_values = list(range(i, i + len(domain_accs)))
        
        # Make sure x and y have the same length
        if len(x_values) != len(domain_accs):
            print(f"Warning: Domain {i} has mismatched dimensions. Adjusting plot.")
            # Use the minimum length
            min_len = min(len(x_values), len(domain_accs))
            x_values = x_values[:min_len]
            domain_accs = domain_accs[:min_len]
        
        plt.plot(x_values, domain_accs, marker='o', label=f"Domain {i}: {domains[i]}", 
                 color=colors[i], linewidth=3, markersize=10, 
                 markeredgecolor='black', markeredgewidth=1.5)
        
        # Annotate last point
        if domain_accs:
            plt.text(x_values[-1], domain_accs[-1], f'{domain_accs[-1]:.2f}', 
                     ha='left', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Domains Trained', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Domain Accuracy After Sequential Training', fontsize=16)
    
    plt.xticks(range(len(domains)), domains, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_forgetting(forgetting, domains, title=None, save_path=None):
    """
    Plot forgetting measure for each domain.
    
    Args:
        forgetting: List where forgetting[i] is the forgetting measure for domain i
        domains: List of domain names
        title: Plot title (optional)
        save_path: Path to save the plot, if None, the plot is displayed
    """
    # Set plot style
    set_plot_style()
    plt.figure()
    
    # Color palette with red-orange gradient for forgetting
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(forgetting)))
    
    # Plot forgetting for each domain except the last one (which has no forgetting)
    domain_indices = range(len(forgetting))
    bars = plt.bar(domain_indices, forgetting, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    # Annotate bars with forgetting values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', 
                 ha='center', va='bottom', 
                 fontweight='bold', fontsize=10)
    
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Forgetting Measure', fontsize=12)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Catastrophic Forgetting by Domain', fontsize=16)
    
    plt.xticks(domain_indices, [domains[i] for i in domain_indices], 
               rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_backward_transfer(bwt, domains, title=None, save_path=None):
    """
    Plot backward transfer for each domain.
    
    Args:
        bwt: List of backward transfer values or single backward transfer value
        domains: List of domain names
        title: Plot title (optional)
        save_path: Path to save the plot, if None, the plot is displayed
    """
    # Set plot style
    set_plot_style()
    plt.figure()
    
    # Handle different input types
    if not isinstance(bwt, list):
        bwt = [bwt]
    
    # Convert to float list
    try:
        bwt = [float(x) for x in bwt]
    except (TypeError, ValueError):
        print("Warning: Could not convert backward transfer values to floats. Using zeros.")
        bwt = [0.0] * len(domains)
    
    # Ensure domains list matches bwt length
    if len(domains) < len(bwt):
        domains = domains + [f'Domain {i}' for i in range(len(domains), len(bwt))]
    elif len(domains) > len(bwt):
        domains = domains[:len(bwt)]
    
    # Color selection based on positive/negative values
    colors = ['green' if x >= 0 else 'red' for x in bwt]
    
    # Plot backward transfer
    bars = plt.bar(range(len(bwt)), bwt, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    # Annotate bars with BWT values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', 
                 ha='center', 
                 va='bottom' if height >= 0 else 'top', 
                 fontweight='bold', fontsize=10)
    
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Backward Transfer', fontsize=12)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Backward Transfer by Domain', fontsize=16)
    
    # Use the truncated domains list
    plt.xticks(range(len(bwt)), domains, 
               rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(confusion_matrix, labels, title=None, save_path=None):
    """Plot confusion matrix for a specific domain."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title or 'Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_learning_curves(training_losses, validation_accuracies, 
                         domains, save_path=None):
    """Plot learning curves for each domain."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot training loss
    for i, losses in enumerate(training_losses):
        ax1.plot(losses, marker='o', label=f"Domain {i}: {domains[i]}")
    
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    for i, accs in enumerate(validation_accuracies):
        ax2.plot(accs, marker='s', label=f"Domain {i}: {domains[i]}")
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy by Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_parameter_importance(model, top_n=20, save_path=None):
    """Visualize the importance of top parameters in EWC."""
    if not hasattr(model, 'importance') or not model.importance:
        print("No parameter importance information available")
        return
    
    # Collect importance values
    param_names = []
    importance_values = []
    
    for name, imp in model.importance.items():
        # Flatten importance tensor
        flat_imp = imp.view(-1).cpu().numpy()
        
        # Get top values
        top_values = np.sort(flat_imp)[-top_n:]
        
        for val in top_values:
            param_names.append(f"{name[:10]}...")
            importance_values.append(val)
    
    # Sort by importance
    sorted_indices = np.argsort(importance_values)[-top_n:]
    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_values = [importance_values[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_values)), sorted_values, align='center')
    plt.yticks(range(len(sorted_values)), sorted_names)
    plt.xlabel('Importance Value')
    plt.title('Top Parameter Importance in EWC')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
def plot_comparison(strategies, avg_accuracies, avg_forgetting, title=None, save_path=None):
    """Enhanced visualization with more informative design"""
    plt.figure(figsize=(16, 9), dpi=300)
    
    # Professional color palette
    colors = sns.color_palette("Set2", 2)
    
    # Bar width and positions
    x = np.arange(len(strategies))
    width = 0.35
    
    # Accuracy bars
    acc_bars = plt.bar(x - width/2, avg_accuracies, width, 
                       label='Avg. Accuracy', 
                       color=colors[0], 
                       edgecolor='black', 
                       linewidth=1.5, 
                       alpha=0.7)
    
    # Anti-forgetting bars
    forget_bars = plt.bar(x + width/2, [1 - f for f in avg_forgetting], width, 
                          label='1 - Avg. Forgetting', 
                          color=colors[1], 
                          edgecolor='black', 
                          linewidth=1.5, 
                          alpha=0.7)
    
    # Annotate bars with values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', 
                     fontweight='bold', 
                     fontsize=10)
    
    autolabel(acc_bars)
    autolabel(forget_bars)
    
    # Styling
    plt.xlabel('Continual Learning Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(title or 'Comparison of Continual Learning Strategies', 
              fontsize=16, fontweight='bold')
    
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def visualize_experiment_results(results_file, output_dir=None):
    """
    Visualize results from a single experiment.
    
    Args:
        results_file: Path to results JSON file
        output_dir: Directory to save plots, if None, plots are displayed
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Extract data with error handling
    domains = results.get('domains', [])
    accuracies = results.get('accuracies', [])
    
    # Handle missing or malformed data
    if not accuracies:
        print("No accuracy data found. Skipping visualization.")
        return
    
    # Plot accuracy over time
    plot_accuracy_over_time(
        accuracies, 
        domains, 
        title="Accuracy Evolution During Sequential Learning",
        save_path=os.path.join(output_dir, 'accuracy.png') if output_dir else None
    )
    
    # Plot forgetting
    forgetting = results.get('forgetting', [0] * len(domains))
    plot_forgetting(
        forgetting, 
        domains, 
        title="Catastrophic Forgetting by Domain",
        save_path=os.path.join(output_dir, 'forgetting.png') if output_dir else None
    )
    
    # Plot backward transfer with robust handling
    backward_transfer = results.get('backward_transfer', [0] * len(domains))
    plot_backward_transfer(
        backward_transfer, 
        domains, 
        title="Backward Transfer by Domain",
        save_path=os.path.join(output_dir, 'backward_transfer.png') if output_dir else None
    )

def compare_strategies(results_files, strategies=None, output_dir=None):
    """
    Compare results from multiple experiments with different strategies.
    
    Args:
        results_files: List of paths to results JSON files
        strategies: List of strategy names (optional, inferred from filenames if None)
        output_dir: Directory to save plots, if None, plots are displayed
    """
    all_results = []
    
    # Load all results
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Infer strategy names if not provided
    if strategies is None:
        strategies = [os.path.basename(os.path.dirname(file_path)) for file_path in results_files]
    
    # Ensure strategies match the number of results
    if len(strategies) != len(all_results):
        print(f"Warning: Number of strategies ({len(strategies)}) doesn't match number of result files ({len(all_results)})")
        # Use the minimum length
        min_len = min(len(strategies), len(all_results))
        strategies = strategies[:min_len]
        all_results = all_results[:min_len]
    
    # Extract comparison metrics
    avg_accuracies = []
    avg_forgetting = []
    
    for results in all_results:
        # Use avg_accuracy or mean of final accuracies
        if 'avg_accuracy' in results:
            avg_accuracies.append(results['avg_accuracy'])
        elif 'final_accuracies' in results:
            avg_accuracies.append(np.mean(results['final_accuracies']))
        else:
            avg_accuracies.append(0)
        
        # Handle forgetting
        if 'forgetting' in results and results['forgetting']:
            avg_forgetting.append(np.mean(results['forgetting']))
        else:
            avg_forgetting.append(0)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparison
    plot_comparison(
        strategies,
        avg_accuracies,
        avg_forgetting,
        title="Comparison of Continual Learning Strategies",
        save_path=os.path.join(output_dir, 'comparison.png') if output_dir else None
    )
def plot_enhanced_comparison(results_files, strategies=None, output_dir=None):
    """
    Generate enhanced comparison visualizations using the detailed metrics.
    
    Args:
        results_files: List of paths to results JSON files
        strategies: List of strategy names (optional, inferred from filenames if None)
        output_dir: Directory to save plots, if None, the plots are displayed
    """
    all_results = []
    
    # Load all results
    for file_path in results_files:
        try:
            # First try to load detailed metrics if available
            detailed_path = os.path.join(os.path.dirname(file_path), 'detailed_metrics.json')
            if os.path.exists(detailed_path):
                with open(detailed_path, 'r') as f:
                    results = json.load(f)
            else:
                # Fall back to standard results file
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    
            all_results.append(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Infer strategy names if not provided
    if strategies is None:
        strategies = [os.path.basename(os.path.dirname(file_path)) for file_path in results_files]
    
    # Ensure strategies match the number of results
    if len(strategies) != len(all_results):
        print(f"Warning: Number of strategies ({len(strategies)}) doesn't match number of result files ({len(all_results)})")
        # Use the minimum length
        min_len = min(len(strategies), len(all_results))
        strategies = strategies[:min_len]
        all_results = all_results[:min_len]
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic Comparison (Accuracy and Forgetting)
    plot_basic_comparison(strategies, all_results, output_dir)
    
    # 2. Transfer Metrics Comparison
    plot_transfer_comparison(strategies, all_results, output_dir)
    
    # 3. Plasticity-Stability Comparison
    plot_plasticity_stability_comparison(strategies, all_results, output_dir)
    
    # 4. Comprehensive Performance Comparison
    plot_comprehensive_comparison(strategies, all_results, output_dir)
    
    # 5. Final Accuracy Across Domains
    plot_final_accuracy_comparison(strategies, all_results, output_dir)

def plot_basic_comparison(strategies, all_results, output_dir=None):
    """Plot the basic comparison of accuracy and forgetting prevention."""
    # Extract metrics
    avg_accuracies = []
    avg_forgetting = []
    
    for results in all_results:
        # Get average accuracy
        if 'avg_accuracy' in results:
            avg_accuracies.append(results['avg_accuracy'])
        elif 'final_accuracies' in results:
            avg_accuracies.append(np.mean(results['final_accuracies']))
        else:
            avg_accuracies.append(0)
        
        # Get forgetting
        if 'forgetting' in results:
            if isinstance(results['forgetting'], list) and results['forgetting']:
                avg_forgetting.append(1 - np.mean(results['forgetting']))
            else:
                avg_forgetting.append(1)  # No forgetting
        else:
            avg_forgetting.append(1)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, avg_accuracies, width, label='Avg. Accuracy', color='#8dd3c7', edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x + width/2, avg_forgetting, width, label='1 - Avg. Forgetting', color='#fb8072', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Continual Learning Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Basic Comparison: Accuracy vs. Forgetting Prevention', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.ylim(0, 1.05)  # Leave room for labels
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'basic_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_transfer_comparison(strategies, all_results, output_dir=None):
    """Plot comparison of transfer learning metrics."""
    # Extract metrics
    backward_transfer = []
    forward_transfer = []
    
    for results in all_results:
        # Get backward transfer
        if 'backward_transfer' in results:
            if isinstance(results['backward_transfer'], list) and results['backward_transfer']:
                backward_transfer.append(np.mean(results['backward_transfer']))
            else:
                backward_transfer.append(0)
        else:
            backward_transfer.append(0)
        
        # Get forward transfer
        if 'forward_transfer' in results:
            if isinstance(results['forward_transfer'], list) and results['forward_transfer']:
                forward_transfer.append(np.mean(results['forward_transfer']))
            else:
                forward_transfer.append(0)
        else:
            forward_transfer.append(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.35
    
    colors = ['#66c2a5', '#fc8d62']
    bars1 = plt.bar(x - width/2, backward_transfer, width, label='Backward Transfer', color=colors[0], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x + width/2, forward_transfer, width, label='Forward Transfer', color=colors[1], edgecolor='black', linewidth=1.5)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.01 if height >= 0 else height - 0.02,
                    f'{height:.2f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Continual Learning Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Transfer Score', fontsize=12, fontweight='bold')
    plt.title('Knowledge Transfer Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'transfer_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_plasticity_stability_comparison(strategies, all_results, output_dir=None):
    """Plot comparison of plasticity and stability metrics."""
    # Extract metrics
    plasticity = []
    stability = []
    ps_ratio = []
    
    for results in all_results:
        # Get plasticity
        if 'plasticity' in results:
            plasticity.append(results['plasticity'])
        else:
            plasticity.append(0)
        
        # Get stability
        if 'stability' in results:
            stability.append(results['stability'])
        else:
            if 'forgetting' in results and isinstance(results['forgetting'], list) and results['forgetting']:
                stability.append(1 - np.mean(results['forgetting']))
            else:
                stability.append(1)
        
        # Get plasticity-stability ratio
        if 'plasticity_stability_ratio' in results:
            ps_ratio.append(results['plasticity_stability_ratio'])
        else:
            # Calculate if possible
            if plasticity[-1] > 0 and stability[-1] > 0:
                ps_ratio.append(plasticity[-1] / stability[-1])
            else:
                ps_ratio.append(0)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    x = np.arange(len(strategies))
    width = 0.25
    
    colors = ['#8da0cb', '#66c2a5', '#fc8d62']
    bars1 = plt.bar(x - width, plasticity, width, label='Plasticity', color=colors[0], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x, stability, width, label='Stability', color=colors[1], edgecolor='black', linewidth=1.5)
    bars3 = plt.bar(x + width, ps_ratio, width, label='Plasticity-Stability Ratio', color=colors[2], edgecolor='black', linewidth=1.5)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.xlabel('Continual Learning Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Plasticity-Stability Balance', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'plasticity_stability_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comprehensive_comparison(strategies, all_results, output_dir=None):
    """Plot comprehensive comparison with radar chart."""
    # Check if we have sufficient metrics for radar chart
    has_detailed_metrics = any('plasticity' in results for results in all_results)
    
    if not has_detailed_metrics:
        print("Insufficient metrics for comprehensive comparison. Using basic comparison.")
        return
    
    # Extract metrics for radar chart
    metrics_names = ['Accuracy', 'Forgetting Prevention', 'Backward Transfer', 
                    'Forward Transfer', 'Plasticity', 'Stability']
    
    # Normalize values between 0 and 1 for radar chart
    metrics_values = []
    
    for results in all_results:
        # Collect values
        values = []
        
        # Accuracy
        if 'avg_accuracy' in results:
            values.append(results['avg_accuracy'])
        elif 'final_accuracies' in results:
            values.append(np.mean(results['final_accuracies']))
        else:
            values.append(0)
        
        # Forgetting Prevention
        if 'forgetting' in results and isinstance(results['forgetting'], list) and results['forgetting']:
            values.append(1 - np.mean(results['forgetting']))
        else:
            values.append(1)
        
        # Backward Transfer
        if 'backward_transfer' in results and isinstance(results['backward_transfer'], list) and results['backward_transfer']:
            # Scale to [0,1], assuming -0.5 to 0.5 range
            bt = np.mean(results['backward_transfer'])
            values.append((bt + 0.5) / 1.0) 
        else:
            values.append(0.5)  # Neutral
        
        # Forward Transfer
        if 'forward_transfer' in results and isinstance(results['forward_transfer'], list) and results['forward_transfer']:
            # Scale to [0,1], assuming -0.5 to 0.5 range
            ft = np.mean(results['forward_transfer'])
            values.append((ft + 0.5) / 1.0)
        else:
            values.append(0.5)  # Neutral
        
        # Plasticity
        if 'plasticity' in results:
            values.append(results['plasticity'])
        else:
            values.append(0)
        
        # Stability
        if 'stability' in results:
            values.append(results['stability'])
        else:
            if 'forgetting' in results and isinstance(results['forgetting'], list) and results['forgetting']:
                values.append(1 - np.mean(results['forgetting']))
            else:
                values.append(1)
        
        metrics_values.append(values)
    
    # Create radar chart
    plt.figure(figsize=(12, 10))
    
    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up subplot in polar coordinates
    ax = plt.subplot(111, polar=True)
    
    # Add metric labels
    plt.xticks(angles[:-1], metrics_names, fontsize=12)
    
    # Plot each strategy
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for i, (values, strategy) in enumerate(zip(metrics_values, strategies)):
        values = values + values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_final_accuracy_comparison(strategies, all_results, output_dir=None):
    """Plot comparison of final accuracy across domains."""
    # Extract final accuracy for each domain
    all_final_accs = []
    domain_count = 0
    
    for results in all_results:
        if 'final_accuracies' in results and isinstance(results['final_accuracies'], list):
            all_final_accs.append(results['final_accuracies'])
            domain_count = max(domain_count, len(results['final_accuracies']))
        elif 'accuracies' in results and isinstance(results['accuracies'], list):
            # Extract final accuracy from each domain's accuracy list
            final_accs = [accs[-1] if accs else 0 for accs in results['accuracies']]
            all_final_accs.append(final_accs)
            domain_count = max(domain_count, len(final_accs))
        else:
            all_final_accs.append([0] * domain_count if domain_count > 0 else [0])
    
    # Ensure all accuracy lists have the same length
    for i in range(len(all_final_accs)):
        if len(all_final_accs[i]) < domain_count:
            all_final_accs[i] = all_final_accs[i] + [0] * (domain_count - len(all_final_accs[i]))
    
    # Get domain names if available
    domain_names = None
    for results in all_results:
        if 'domains' in results and isinstance(results['domains'], list):
            domain_names = results['domains']
            break
    
    if not domain_names or len(domain_names) != domain_count:
        domain_names = [f"Domain {i+1}" for i in range(domain_count)]
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    x = np.arange(domain_count)
    width = 0.8 / len(strategies)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for i, (strategy, final_accs) in enumerate(zip(strategies, all_final_accs)):
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = plt.bar(x + offset, final_accs, width, label=strategy, color=colors[i], edgecolor='black', linewidth=1)
    
    plt.xlabel('Domain', fontsize=12, fontweight='bold')
    plt.ylabel('Final Accuracy', fontsize=12, fontweight='bold')
    plt.title('Final Accuracy Across Domains', fontsize=16, fontweight='bold')
    plt.xticks(x, domain_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'domain_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results', type=str, help='Path to results JSON file')
    parser.add_argument('--results_dir', type=str, help='Directory containing results JSON files for comparison')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--strategies', type=str, nargs='+', default=None, help='Strategy names for comparison')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.results:
        # Visualize single experiment
        visualize_experiment_results(args.results, args.output_dir)
    
    elif args.results_dir:
        # Find all results files in directory
        results_files = glob.glob(os.path.join(args.results_dir, '**/results.json'), recursive=True)
        
        if not results_files:
            print(f"No results files found in {args.results_dir}")
            return
        
        # Compare strategies
        compare_strategies(results_files, args.strategies, args.output_dir)
    
    else:
        print("Please provide either --results or --results_dir")

if __name__ == "__main__":
    main()