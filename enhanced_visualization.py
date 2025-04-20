# enhanced_visualization.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd

def set_plot_style():
    """Set consistent plot styling."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.figsize': (14, 8),
    })

def load_results(results_dir):
    print(f"Searching for results in: {results_dir}")
    
    # Updated to find all results.json files
    results_files = glob.glob(os.path.join(results_dir, '**', 'results.json'), recursive=True)
    detailed_files = glob.glob(os.path.join(results_dir, '**', 'detailed_metrics.json'), recursive=True)
    
    # Debug prints
    print(f"Total results files found: {len(results_files)}")
    print(f"Total detailed metrics files found: {len(detailed_files)}")
    
    for file in results_files:
        print(f"Results file: {file}")
    
    for file in detailed_files:
        print(f"Detailed metrics file: {file}")
    
    # Create paired files dictionary
    paired_files = {}
    for results_file in results_files:
        exp_dir = os.path.dirname(results_file)
        detailed_file = os.path.join(exp_dir, 'detailed_metrics.json')
        
        # Check if detailed file exists
        if os.path.exists(detailed_file):
            paired_files[exp_dir] = {
                'results': results_file,
                'detailed': detailed_file
            }
        else:
            paired_files[exp_dir] = {
                'results': results_file,
                'detailed': None
            }
    
    # Load all data
    all_data = {}
    for exp_dir, files in paired_files.items():
        # Extract experiment name (modify as needed)
        exp_name = os.path.basename(exp_dir)
        
        # Load results
        try:
            with open(files['results'], 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading results from {files['results']}: {e}")
            continue
        
        # Load detailed metrics if available
        detailed = None
        if files['detailed']:
            try:
                with open(files['detailed'], 'r') as f:
                    detailed = json.load(f)
            except Exception as e:
                print(f"Error loading detailed metrics from {files['detailed']}: {e}")
        
        all_data[exp_name] = {
            'results': results,
            'detailed': detailed,
            'dir': exp_dir
        }
    
    print(f"Total experiments processed: {len(all_data)}")
    return all_data

def extract_strategies(all_data):
    """Extract strategy names from experiment data."""
    strategies = []
    
    for exp_name, data in all_data.items():
        # Try to parse EWC and replay values from the name
        import re
        match = re.search(r'ewc(\d+\.?\d*)_replay(\d+)', os.path.basename(data['dir']))
        if match:
            ewc = match.group(1)
            replay = match.group(2)
            strategy = f"EWC={ewc},RP={replay}"
        else:
            strategy = exp_name
        
        strategies.append(strategy)
    
    return strategies

def create_comprehensive_report(results_dir, output_dir):
    """Create a comprehensive report with all metrics visualized."""
    print("Starting comprehensive analysis...")
    
    # Add extensive logging
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results with detailed logging
    all_data = load_results(results_dir)
    
    # Log loaded data
    print(f"Total experiments loaded: {len(all_data)}")
    for name, data in all_data.items():
        print(f"Experiment: {name}")
        print(f"  Results file exists: {bool(data['results'])}")
        print(f"  Detailed metrics exists: {bool(data['detailed'])}")
    
    # Extract strategies
    strategies = extract_strategies(all_data)
    print(f"Extracted strategies: {strategies}")

    # Add try-except to catch and print any errors during visualization
    try:
        # Visualization functions
        plot_basic_performance(all_data, strategies, output_dir)
        plot_forgetting_metrics(all_data, strategies, output_dir)
        plot_transfer_metrics(all_data, strategies, output_dir)
        plot_plasticity_stability_metrics(all_data, strategies, output_dir)
        plot_resource_metrics(all_data, strategies, output_dir)
        plot_domain_performance(all_data, strategies, output_dir)
        plot_overall_comparison(all_data, strategies, output_dir)
        
        # Generate summary table and find best config
        df = generate_metrics_table(all_data, strategies, output_dir)
        find_best_config(all_data, strategies, output_dir)
        
        print("Comprehensive analysis completed successfully!")
    except Exception as e:
        print(f"Error during comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

def plot_basic_performance(all_data, strategies, output_dir):
    """Plot basic performance metrics."""
    # Extract metrics
    avg_accuracies = []
    avg_f1_scores = []
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
        # Get average accuracy
        if 'avg_accuracy' in results:
            avg_accuracies.append(results['avg_accuracy'])
        elif 'final_accuracies' in results:
            avg_accuracies.append(np.mean(results['final_accuracies']))
        else:
            avg_accuracies.append(0)
        
        # Get F1 scores if available
        if detailed and 'avg_f1_score' in detailed:
            avg_f1_scores.append(detailed['avg_f1_score'])
        else:
            avg_f1_scores.append(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, avg_accuracies, width, label='Avg. Accuracy', 
                   color='#1f77b4', edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x + width/2, avg_f1_scores, width, label='Avg. F1 Score', 
                   color='#ff7f0e', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Continual Learning Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Basic Performance Metrics', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'basic_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_forgetting_metrics(all_data, strategies, output_dir):
    """Plot forgetting metrics."""
    # Extract metrics
    avg_forgetting = []
    max_forgetting = []
    cf_events = []  # Catastrophic forgetting events
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
        # Average forgetting
        if 'forgetting' in results and isinstance(results['forgetting'], list) and results['forgetting']:
            avg_forgetting.append(np.mean(results['forgetting']))
        else:
            avg_forgetting.append(0)
        
        # Maximum forgetting
        if detailed and 'max_forgetting' in detailed:
            max_forgetting.append(detailed['max_forgetting'])
        elif 'forgetting' in results and isinstance(results['forgetting'], list) and results['forgetting']:
            max_forgetting.append(max(results['forgetting']))
        else:
            max_forgetting.append(0)
        
        # Catastrophic forgetting events
        if detailed and 'catastrophic_forgetting_events' in detailed:
            cf_events.append(detailed['catastrophic_forgetting_events'])
        else:
            cf_events.append(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.25
    
    bars1 = plt.bar(x - width, avg_forgetting, width, label='Avg. Forgetting', 
                   color='#d62728', edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x, max_forgetting, width, label='Max Forgetting', 
                   color='#9467bd', edgecolor='black', linewidth=1.5)
    bars3 = plt.bar(x + width, cf_events, width, label='Catastrophic Events', 
                   color='#8c564b', edgecolor='black', linewidth=1.5)
    
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
    plt.ylabel('Forgetting Metrics', fontsize=12, fontweight='bold')
    plt.title('Forgetting Metrics Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'forgetting_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_transfer_metrics(all_data, strategies, output_dir):
    """Plot transfer learning metrics."""
    # Extract metrics
    backward_transfer = []
    forward_transfer = []
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
        # Backward transfer
        if detailed and 'avg_backward_transfer' in detailed:
            backward_transfer.append(detailed['avg_backward_transfer'])
        elif 'backward_transfer' in results and isinstance(results['backward_transfer'], list) and results['backward_transfer']:
            backward_transfer.append(np.mean(results['backward_transfer']))
        else:
            backward_transfer.append(0)
        
        # Forward transfer
        if detailed and 'avg_forward_transfer' in detailed:
            forward_transfer.append(detailed['avg_forward_transfer'])
        elif 'forward_transfer' in results and isinstance(results['forward_transfer'], list) and results['forward_transfer']:
            forward_transfer.append(np.mean(results['forward_transfer']))
        else:
            forward_transfer.append(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.35
    
    colors = ['#66c2a5', '#fc8d62']
    bars1 = plt.bar(x - width/2, backward_transfer, width, label='Backward Transfer', 
                   color=colors[0], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x + width/2, forward_transfer, width, label='Forward Transfer', 
                   color=colors[1], edgecolor='black', linewidth=1.5)
    
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
    
    plt.savefig(os.path.join(output_dir, 'transfer_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_plasticity_stability_metrics(all_data, strategies, output_dir):
    """Plot plasticity and stability metrics."""
    # Extract metrics
    plasticity = []
    stability = []
    ps_ratio = []
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
        # Plasticity
        if detailed and 'plasticity' in detailed:
            plasticity.append(detailed['plasticity'])
        elif 'plasticity' in results:
            plasticity.append(results['plasticity'])
        else:
            plasticity.append(0)
        
        # Stability
        if detailed and 'stability' in detailed:
            stability.append(detailed['stability'])
        elif 'stability' in results:
            stability.append(results['stability'])
        else:
            stability.append(0)
        
        # Plasticity-Stability Ratio
        if detailed and 'plasticity_stability_ratio' in detailed:
            ps_ratio.append(detailed['plasticity_stability_ratio'])
        elif 'plasticity_stability_ratio' in results:
            ps_ratio.append(results['plasticity_stability_ratio'])
        else:
            ps_ratio.append(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.25
    
    colors = ['#8da0cb', '#66c2a5', '#fc8d62']
    bars1 = plt.bar(x - width, plasticity, width, label='Plasticity', 
                   color=colors[0], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x, stability, width, label='Stability', 
                   color=colors[1], edgecolor='black', linewidth=1.5)
    bars3 = plt.bar(x + width, ps_ratio, width, label='Plasticity-Stability Ratio', 
                   color=colors[2], edgecolor='black', linewidth=1.5)
    
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
    plt.title('Plasticity-Stability Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'plasticity_stability_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_resource_metrics(all_data, strategies, output_dir):
    """Plot resource utilization metrics."""
    # Extract metrics
    buffer_utilization = []
    ewc_overhead = []
    training_time = []
    
    for strategy, data in zip(strategies, all_data.values()):
        detailed = data['detailed']
        
        # Check if detailed metrics exist
        if not detailed:
            buffer_utilization.append(0)
            ewc_overhead.append(0)
            training_time.append(0)
            continue
        
        # Buffer utilization
        if 'replay_efficiency' in detailed and 'buffer_utilization' in detailed['replay_efficiency']:
            buffer_utilization.append(detailed['replay_efficiency']['buffer_utilization'])
        else:
            buffer_utilization.append(0)
        
        # EWC overhead
        if 'training_overhead' in detailed and 'ewc_overhead_ratio' in detailed['training_overhead']:
            ewc_overhead.append(detailed['training_overhead']['ewc_overhead_ratio'])
        else:
            ewc_overhead.append(0)
        
        # Training time
        if 'training_overhead' in detailed and 'avg_train_time' in detailed['training_overhead']:
            training_time.append(detailed['training_overhead']['avg_train_time'])
        else:
            training_time.append(0)
    
    # Normalize training time for better visualization
    if max(training_time) > 0:
        training_time = [t / max(training_time) for t in training_time]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(strategies))
    width = 0.25
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    bars1 = plt.bar(x - width, buffer_utilization, width, label='Buffer Utilization', 
                   color=colors[0], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x, ewc_overhead, width, label='EWC Overhead Ratio', 
                   color=colors[1], edgecolor='black', linewidth=1.5)
    bars3 = plt.bar(x + width, training_time, width, label='Normalized Training Time', 
                   color=colors[2], edgecolor='black', linewidth=1.5)
    
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
    plt.ylabel('Resource Metrics', fontsize=12, fontweight='bold')
    plt.title('Resource Utilization Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'resource_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_performance(all_data, strategies, output_dir):
    """Plot performance across domains."""
    # First, determine common domains across experiments
    all_domains = set()
    for data in all_data.values():
        results = data['results']
        if 'domains' in results:
            all_domains.update(results['domains'])
    
    all_domains = sorted(list(all_domains))
    
    if not all_domains:
        print("No domain information found. Skipping domain performance visualization.")
        return
    
    # Extract final accuracies for each domain
    domain_accuracies = {domain: [] for domain in all_domains}
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        
        if 'final_accuracies' in results and 'domains' in results:
            domains = results['domains']
            accs = results['final_accuracies']
            
            # Match accuracies to common domains
            for i, domain in enumerate(domains):
                if i < len(accs) and domain in domain_accuracies:
                    domain_accuracies[domain].append(accs[i])
                else:
                    domain_accuracies[domain].append(0)
        else:
            # Fill with zeros if data not available
            for domain in all_domains:
                domain_accuracies[domain].append(0)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Set up bar positions
    num_domains = len(all_domains)
    num_strategies = len(strategies)
    bar_width = 0.8 / num_strategies
    
    # Setup colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    
    # Plot bars for each strategy across domains
    for i, strategy in enumerate(strategies):
        domain_values = [domain_accuracies[domain][i] for domain in all_domains]
        x = np.arange(num_domains)
        offset = (i - num_strategies/2 + 0.5) * bar_width
        plt.bar(x + offset, domain_values, bar_width, label=strategy, color=colors[i], edgecolor='black', linewidth=1)
    
    plt.xlabel('Domain', fontsize=12, fontweight='bold')
    plt.ylabel('Final Accuracy', fontsize=12, fontweight='bold')
    plt.title('Final Accuracy Across Domains', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(num_domains), all_domains, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'domain_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_overall_comparison(all_data, strategies, output_dir):
    """Create radar chart with comprehensive comparison."""
    # Extract metrics for radar chart
    metrics_names = ['Accuracy', 'Forgetting Prevention', 'Backward Transfer', 
                    'Forward Transfer', 'Plasticity', 'Stability']
    
    metrics_values = []
    
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
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
        if detailed and 'avg_backward_transfer' in detailed:
            # Scale to [0,1], assuming -0.5 to 0.5 range
            bt = detailed['avg_backward_transfer']
            values.append((bt + 0.5) / 1.0) 
        elif 'backward_transfer' in results and isinstance(results['backward_transfer'], list) and results['backward_transfer']:
            bt = np.mean(results['backward_transfer'])
            values.append((bt + 0.5) / 1.0)
        else:
            values.append(0.5)
        
        # Forward Transfer
        if detailed and 'avg_forward_transfer' in detailed:
            # Scale to [0,1], assuming -0.5 to 0.5 range
            ft = detailed['avg_forward_transfer']
            values.append((ft + 0.5) / 1.0)
        elif 'forward_transfer' in results and isinstance(results['forward_transfer'], list) and results['forward_transfer']:
            ft = np.mean(results['forward_transfer'])
            values.append((ft + 0.5) / 1.0)
        else:
            values.append(0.5)
        
        # Plasticity
        if detailed and 'plasticity' in detailed:
            values.append(detailed['plasticity'])
        elif 'plasticity' in results:
            values.append(results['plasticity'])
        else:
            values.append(0)
        
        # Stability
        if detailed and 'stability' in detailed:
            values.append(detailed['stability'])
        elif 'stability' in results:
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
    
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_metrics_table(all_data, strategies, output_dir):
    """Generate a summary table with all metrics."""
    # Define all metrics to include
    metric_groups = {
        'Performance': ['avg_accuracy', 'avg_f1_score'],
        'Forgetting': ['avg_forgetting', 'max_forgetting', 'catastrophic_forgetting_events'],
        'Transfer': ['avg_backward_transfer', 'avg_forward_transfer'],
        'Plasticity-Stability': ['plasticity', 'stability', 'plasticity_stability_ratio'],
        'Resource': ['buffer_utilization', 'ewc_overhead_ratio']
    }
    
    # Flatten metrics list
    all_metrics = []
    for group, metrics in metric_groups.items():
        all_metrics.extend([(group, m) for m in metrics])
    
    # Create DataFrame
    metrics_data = []
    for strategy, data in zip(strategies, all_data.values()):
        results = data['results']
        detailed = data['detailed']
        
        row = {'Strategy': strategy}
        
        # Extract metrics
        for group, metric in all_metrics:
            # Handle metrics in results
            if metric in results:
                row[f"{group}: {metric}"] = results[metric]
            
            # Handle metrics in detailed metrics
            elif detailed:
                # Direct metrics
                if metric in detailed:
                    row[f"{group}: {metric}"] = detailed[metric]
                
                # Nested metrics
                else:
                    for key, value in detailed.items():
                        if isinstance(value, dict) and metric in value:
                            row[f"{group}: {metric}"] = value[metric]
                            break
            
            # Handle derived metrics
            elif metric == 'avg_forgetting' and 'forgetting' in results:
                if isinstance(results['forgetting'], list) and results['forgetting']:
                    row[f"{group}: {metric}"] = np.mean(results['forgetting'])
            
            # Default to 0 if not found
            if f"{group}: {metric}" not in row:
                row[f"{group}: {metric}"] = 0
        
        metrics_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Save to markdown for better readability
    markdown_path = os.path.join(output_dir, 'metrics_summary.md')
    with open(markdown_path, 'w') as f:
        f.write(df.to_markdown(index=False))
    
    return df

def find_best_config(all_data, strategies, output_dir):
    """Find the best configuration based on weighted metrics."""
    # Load metrics from the summary table
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Generate metrics table if it doesn't exist
        df = generate_metrics_table(all_data, strategies, output_dir)
    
    # Define weights for different metrics
    # Higher weights indicate more importance
    weights = {
        'Performance: avg_accuracy': 1.0,
        'Performance: avg_f1_score': 0.8,
        'Forgetting: avg_forgetting': -1.0,  # Negative weight as lower is better
        'Forgetting: max_forgetting': -0.7,  # Negative weight as lower is better
        'Forgetting: catastrophic_forgetting_events': -0.9,  # Negative weight as lower is better
        'Transfer: avg_backward_transfer': 0.6,
        'Transfer: avg_forward_transfer': 0.6,
        'Plasticity-Stability: plasticity': 0.8,
        'Plasticity-Stability: stability': 0.8,
        'Plasticity-Stability: plasticity_stability_ratio': 0.5,
        'Resource: buffer_utilization': 0.3,
        'Resource: ewc_overhead_ratio': -0.3  # Negative weight as lower is better
    }
    
    # Calculate weighted scores
    weighted_scores = {}
    
    for strategy in strategies:
        strategy_row = df[df['Strategy'] == strategy]
        
        if strategy_row.empty:
            weighted_scores[strategy] = 0
            continue
        
        total_score = 0
        for metric, weight in weights.items():
            if metric in strategy_row.columns:
                value = strategy_row[metric].values[0]
                if not pd.isna(value) and isinstance(value, (int, float)):
                    total_score += value * weight
        
        weighted_scores[strategy] = total_score
    
    # Find best strategy
    best_strategy = max(weighted_scores.items(), key=lambda x: x[1])
    
    # Create summary report
    summary = {
        'weighted_scores': weighted_scores,
        'best_strategy': best_strategy[0],
        'best_score': best_strategy[1],
        'weights': weights
    }
    
    # Save results
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualization of weighted scores
    plt.figure(figsize=(14, 8))
    strategies_list = list(weighted_scores.keys())
    scores = list(weighted_scores.values())
    
    bars = plt.bar(strategies_list, scores, color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Highlight best strategy
    best_index = strategies_list.index(best_strategy[0])
    bars[best_index].set_color('green')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Weighted Score', fontsize=12, fontweight='bold')
    plt.title('Strategy Ranking (Higher is Better)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'strategy_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary text file with an explanation
    with open(os.path.join(output_dir, 'best_config_explanation.txt'), 'w') as f:
        f.write("BEST CONFIGURATION ANALYSIS\n")
        f.write("==========================\n\n")
        f.write(f"Best Strategy: {best_strategy[0]} (Score: {best_strategy[1]:.2f})\n\n")
        f.write("Metric Weights Used:\n")
        for metric, weight in weights.items():
            f.write(f"  {metric}: {weight}\n")
        
        f.write("\nAll Strategy Scores (ranked):\n")
        for strategy, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {strategy}: {score:.2f}\n")
        
        f.write("\nAnalysis:\n")
        f.write("The weighted scoring system balances performance metrics (accuracy, F1 score),\n")
        f.write("forgetting resistance (average and maximum forgetting, catastrophic events),\n")
        f.write("transfer learning capabilities (backward and forward transfer),\n")
        f.write("plasticity-stability trade-off, and resource efficiency.\n\n")
        
        f.write("The best configuration achieves the optimal balance across these dimensions,\n")
        f.write("with particular emphasis on maintaining high accuracy while minimizing forgetting.\n")
    
    return best_strategy[0]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive metrics visualizations')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='comprehensive_analysis', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    create_comprehensive_report(args.results_dir, args.output_dir)
    print(f"Comprehensive analysis completed. Results saved to {args.output_dir}")