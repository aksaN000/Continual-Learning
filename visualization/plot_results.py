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