"""
Main script for the continual learning project.
This script serves as the entry point for the entire pipeline.
"""

import argparse
import os
import glob
import enhanced_visualization

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Continual Learning for Text Command Understanding'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data preparation command
    data_parser = subparsers.add_parser('prepare-data', help='Download and prepare data')
    
    # Run experiment command
    exp_parser = subparsers.add_parser('run-experiment', help='Run continual learning experiment')
    exp_parser.add_argument('--config', type=str, default=None, help='Path to config file')
    exp_parser.add_argument('--replay_buffer_size', type=int, default=None, help='Replay buffer capacity')
    exp_parser.add_argument('--replay_batch_size', type=int, default=None, help='Replay batch size')
    exp_parser.add_argument('--ewc_lambda', type=float, default=None, help='EWC regularization strength')
    exp_parser.add_argument('--use_ewc', action='store_true', help='Use EWC regularization')
    exp_parser.add_argument('--no_ewc', action='store_true', help='Disable EWC regularization')
    exp_parser.add_argument('--use_replay', action='store_true', help='Use experience replay')
    exp_parser.add_argument('--no_replay', action='store_true', help='Disable experience replay')
    exp_parser.add_argument('--epochs', type=int, default=None, help='Epochs per domain')
    exp_parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    exp_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    exp_parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    # Compare strategies command
    compare_parser = subparsers.add_parser('compare', help='Compare different continual learning strategies')
    compare_parser.add_argument('--results_dir', type=str, required=True, help='Directory containing results')
    compare_parser.add_argument('--output_dir', type=str, default='comparison_results', help='Directory to save plots')
    compare_parser.add_argument('--strategies', type=str, nargs='+', default=None, help='Strategy names')
    
    # Visualize results command
    vis_parser = subparsers.add_parser('visualize', help='Visualize experiment results')
    vis_parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    vis_parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots')
    
    # Run batched experiments command
    batch_parser = subparsers.add_parser('batch', help='Run multiple experiments with different configurations')
    batch_parser.add_argument('--ewc_values', type=float, nargs='+', default=[0, 0.1, 1, 10], 
                             help='EWC lambda values to test')
    batch_parser.add_argument('--replay_sizes', type=int, nargs='+', default=[0, 100, 500, 1000], 
                             help='Replay buffer sizes to test')
    batch_parser.add_argument('--base_config', type=str, default=None, 
                             help='Base configuration file')
    batch_parser.add_argument('--output_dir', type=str, default='batch_results', 
                             help='Directory to save results')
    
    # Comprehensive analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Generate comprehensive analysis of all metrics')
    analyze_parser.add_argument('--results_dir', type=str, required=True, help='Directory containing experiment results')
    analyze_parser.add_argument('--output_dir', type=str, default='comprehensive_analysis', help='Directory to save visualizations')
    
    return parser.parse_args()

def run_data_preparation():
    """Run data preparation."""
    from data.download_data import download_hwu64
    download_hwu64()

def run_experiment(args):
    """Run a single experiment."""
    from experiments.run_experiment import main as run_experiment_main
    # Replace sys.argv with our args
    import sys
    old_argv = sys.argv
    sys.argv = ['run_experiment.py']
    
    # Add arguments
    if args.config:
        sys.argv.extend(['--config', args.config])
    if args.replay_buffer_size is not None:
        sys.argv.extend(['--replay_buffer_size', str(args.replay_buffer_size)])
    if args.replay_batch_size is not None:
        sys.argv.extend(['--replay_batch_size', str(args.replay_batch_size)])
    if args.ewc_lambda is not None:
        sys.argv.extend(['--ewc_lambda', str(args.ewc_lambda)])
    if args.use_ewc:
        sys.argv.append('--use_ewc')
    if args.no_ewc:
        sys.argv.append('--no_ewc')
    if args.use_replay:
        sys.argv.append('--use_replay')
    if args.no_replay:
        sys.argv.append('--no_replay')
    if args.epochs is not None:
        sys.argv.extend(['--epochs', str(args.epochs)])
    if args.learning_rate is not None:
        sys.argv.extend(['--learning_rate', str(args.learning_rate)])
    if args.seed is not None:
        sys.argv.extend(['--seed', str(args.seed)])
    if args.name is not None:
        sys.argv.extend(['--name', args.name])
    
    # Run experiment
    run_experiment_main()
    
    # Restore sys.argv
    sys.argv = old_argv

def compare_strategies(args):
    """Compare different continual learning strategies."""
    from visualization.plot_results import plot_enhanced_comparison
    import glob
    
    results_files = glob.glob(os.path.join(args.results_dir, '**/results.json'), recursive=True)
    
    if not results_files:
        print(f"No results files found in {args.results_dir}")
        return
    
    # Extract strategy names from directory names if not provided
    strategies = args.strategies
    if not strategies:
        strategies = [os.path.basename(os.path.dirname(file_path)) for file_path in results_files]
    
    # Use enhanced comparison
    plot_enhanced_comparison(results_files, strategies, args.output_dir)
    print(f"Enhanced comparison visualizations saved to {args.output_dir}")

def visualize_results(args):
    """Visualize experiment results."""
    from visualization.plot_results import visualize_experiment_results
    
    visualize_experiment_results(args.results, args.output_dir)

def run_batch_experiments(args):
    """Run multiple experiments with different configurations."""
    from experiments.configs.default_config import config as default_config
    from experiments.run_experiment import run_experiment
    import json
    import copy
    
    # Load base config
    config = copy.deepcopy(default_config)
    if args.base_config:
        with open(args.base_config, 'r') as f:
            base_config = json.load(f)
            config.update(base_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments with different EWC values
    for ewc_lambda in args.ewc_values:
        # Skip EWC = 0 for non-zero replay sizes to avoid duplication
        if ewc_lambda == 0 and 0 not in args.replay_sizes:
            args.replay_sizes.insert(0, 0)
            
        for replay_size in args.replay_sizes:
            # Configure experiment
            exp_config = copy.deepcopy(config)
            
            # Set EWC
            exp_config['experiment']['use_ewc'] = (ewc_lambda > 0)
            exp_config['continual']['ewc_lambda'] = ewc_lambda
            
            # Set replay
            exp_config['experiment']['use_replay'] = (replay_size > 0)
            exp_config['continual']['replay_buffer_size'] = replay_size
            
            # Set name
            exp_config['experiment']['name'] = f"ewc{ewc_lambda}_replay{replay_size}"
            
            # Run experiment
            print(f"\n{'='*50}")
            print(f"Running experiment: EWC={ewc_lambda}, Replay={replay_size}")
            print(f"{'='*50}\n")
            
            # Save experiment results in the batch output directory
            exp_config['output']['save_dir'] = args.output_dir
            
            # Run experiment
            run_experiment(exp_config)
    
    # Compare all strategies
    results_files = glob.glob(os.path.join(args.output_dir, '**/results.json'), recursive=True)
    
    if results_files:
        from visualization.plot_results import compare_strategies
        
        # Extract strategy names from directories
        import re
        strategies = []
        for file_path in results_files:
            match = re.search(r'ewc(\d+\.?\d*)_replay(\d+)', os.path.dirname(file_path))
            if match:
                ewc = match.group(1)
                replay = match.group(2)
                strategies.append(f"EWC={ewc},RP={replay}")
            else:
                strategies.append(os.path.basename(os.path.dirname(file_path)))
        
        # Create comparison plots
        compare_strategies(
            results_files, 
            strategies, 
            os.path.join(args.output_dir, 'comparison')
        )

def run_comprehensive_analysis(args):
    """Run comprehensive analysis on experiment results."""
    from enhanced_visualization import create_comprehensive_report
    create_comprehensive_report(args.results_dir, args.output_dir)
    print(f"Comprehensive analysis completed. Results saved to {args.output_dir}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # For any command other than prepare-data, first check if data exists
    if args.command != 'prepare-data':
        if not os.path.exists('data/processed') or not os.listdir('data/processed'):
            print("Dataset not found. Running data preparation first...")
            run_data_preparation()
    
    if args.command == 'prepare-data':
        run_data_preparation()
    elif args.command == 'run-experiment':
        run_experiment(args)
    elif args.command == 'compare':
        compare_strategies(args)
    elif args.command == 'visualize':
        visualize_results(args)
    elif args.command == 'batch':
        run_batch_experiments(args)
    elif args.command == 'analyze':
        run_comprehensive_analysis(args)
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == "__main__":
    main()