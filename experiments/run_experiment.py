import os
import argparse
import json
import torch
import random
import numpy as np
from datetime import datetime
from transformers import BertTokenizer

# Import configurations
from experiments.configs.default_config import config as default_config

# Import data utilities
from data.data_utils import load_domain_data, get_dataloader, get_num_labels_per_domain

# Import model
from models import ContinualTextCommandLearner

# Import training utilities
from training import ReplayBuffer, train_continual_learning

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_domains(data_dir):
    """Load the list of domains from the domains.txt file."""
    domains_file = os.path.join(data_dir, "domains.txt")
    if os.path.exists(domains_file):
        with open(domains_file, 'r') as f:
            domains = [line.strip() for line in f.readlines()]
        return domains
    else:
        # If domains.txt doesn't exist, try to infer from directory structure
        domains = []
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)) and item != "__pycache__":
                domains.append(item)
        return domains

def prepare_data_loaders(domains, tokenizer, config):
    """
    Prepare data loaders for all domains.
    
    Args:
        domains: List of domain names
        tokenizer: Tokenizer for processing text
        config: Configuration dictionary
        
    Returns:
        data_loaders: Dictionary mapping domain names to (train_loader, val_loader) tuples
    """
    data_loaders = {}
    
    for domain in domains:
        print(f"Loading data for domain: {domain}")
        
        # Load train data
        train_texts, train_labels = load_domain_data(
            domain, 
            split='train', 
            data_dir=config['data']['data_dir']
        )
        
        # Load validation data
        val_texts, val_labels = load_domain_data(
            domain, 
            split='val', 
            data_dir=config['data']['data_dir']
        )
        
        # Create data loaders
        train_loader = get_dataloader(
            train_texts, 
            train_labels, 
            tokenizer, 
            batch_size=config['data']['batch_size'], 
            shuffle=True
        )
        
        val_loader = get_dataloader(
            val_texts, 
            val_labels, 
            tokenizer, 
            batch_size=config['data']['batch_size'], 
            shuffle=False
        )
        
        data_loaders[domain] = (train_loader, val_loader)
        
        print(f"  Train examples: {len(train_texts)}")
        print(f"  Validation examples: {len(val_texts)}")
    
    return data_loaders

def run_experiment(config):
    """
    Run a continual learning experiment with the given configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Set up experiment
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['output']['save_dir'], f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load domains
    domains = load_domains(config['data']['data_dir'])
    print(f"Domains: {domains}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['base_model'])
    
    # Prepare data loaders
    data_loaders = prepare_data_loaders(domains, tokenizer, config)
    
    # Get number of labels per domain
    num_labels_per_domain = get_num_labels_per_domain(domains, config['data']['data_dir'])
    print(f"Number of labels per domain: {num_labels_per_domain}")
    
    # Initialize model
    model = ContinualTextCommandLearner(num_labels_per_domain, config['model']['base_model'])
    model.to(device)
    
    # Initialize replay buffer
    replay_buffer_size = config['continual']['replay_buffer_size'] if config['experiment']['use_replay'] else 0
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # Calculate appropriate EWC lambda based on experiment settings
    ewc_lambda = 0.0
    if config['experiment']['use_ewc']:
        base_ewc_lambda = config['continual']['ewc_lambda']
        
        # If both EWC and replay are active, adjust the EWC lambda based on the replay settings
        if config['experiment']['use_replay']:
            # Increase EWC for larger models to balance with replay
            ewc_lambda = base_ewc_lambda * 1.2  # Slightly stronger EWC when combined with replay
        else:
            ewc_lambda = base_ewc_lambda
        
        model.ewc_lambda = ewc_lambda
    else:
        model.ewc_lambda = 0.0  # Explicitly set to 0 to disable EWC
    
    # Set replay batch size with adaptive adjustment
    replay_batch_size = 0
    if config['experiment']['use_replay']:
        base_replay_batch_size = config['continual']['replay_batch_size']
        
        # If both EWC and replay are active, adjust the replay batch size
        if config['experiment']['use_ewc']:
            # Slightly increase replay batch size to balance with EWC
            replay_batch_size = base_replay_batch_size
        else:
            replay_batch_size = base_replay_batch_size
    
    # Log the settings being used
    print(f"\nContinual Learning Settings:")
    print(f"  EWC: {'Enabled' if config['experiment']['use_ewc'] else 'Disabled'}")
    if config['experiment']['use_ewc']:
        print(f"  EWC Lambda: {model.ewc_lambda}")
    print(f"  Replay: {'Enabled' if config['experiment']['use_replay'] else 'Disabled'}")
    if config['experiment']['use_replay']:
        print(f"  Replay Buffer Size: {replay_buffer_size}")
        print(f"  Replay Batch Size: {replay_batch_size}")
    print(f"  Epochs per domain: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    
    # Run continual learning
    print("\nStarting continual learning training...")
    metrics = train_continual_learning(
        model,
        domains,
        data_loaders,
        replay_buffer,
        device,
        epochs=config['training']['epochs'],
        replay_batch_size=replay_batch_size,
        learning_rate=config['training']['learning_rate']
    )
    
    # Save results
    results = {
        'domains': domains,
        'accuracies': [accs for accs in metrics.domain_accs],
        'forgetting': metrics.compute_forgetting(),
        'backward_transfer': metrics.compute_backward_transfer(),
        'final_accuracies': [accs[-1] if accs else 0 for accs in metrics.domain_accs],
        'avg_accuracy': metrics.compute_avg_accuracy()
    }
    
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model if specified
    if config['output']['save_model']:
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model.pt'))
    
    print(f"\nExperiment completed. Results saved to {experiment_dir}")
    
    return results, metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run continual learning experiment')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--replay_buffer_size', type=int, default=None, help='Replay buffer capacity')
    parser.add_argument('--replay_batch_size', type=int, default=None, help='Replay batch size')
    parser.add_argument('--ewc_lambda', type=float, default=None, help='EWC regularization strength')
    parser.add_argument('--use_ewc', action='store_true', help='Use EWC regularization')
    parser.add_argument('--no_ewc', action='store_true', help='Disable EWC regularization')
    parser.add_argument('--use_replay', action='store_true', help='Use experience replay')
    parser.add_argument('--no_replay', action='store_true', help='Disable experience replay')
    parser.add_argument('--epochs', type=int, default=None, help='Epochs per domain')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load default config
    config = default_config.copy()
    
    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Override with command line arguments
    if args.replay_buffer_size is not None:
        config['continual']['replay_buffer_size'] = args.replay_buffer_size
    if args.replay_batch_size is not None:
        config['continual']['replay_batch_size'] = args.replay_batch_size
    if args.ewc_lambda is not None:
        config['continual']['ewc_lambda'] = args.ewc_lambda
    if args.use_ewc:
        config['experiment']['use_ewc'] = True
    if args.no_ewc:
        config['experiment']['use_ewc'] = False
    if args.use_replay:
        config['experiment']['use_replay'] = True
    if args.no_replay:
        config['experiment']['use_replay'] = False
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.name is not None:
        config['experiment']['name'] = args.name
    
    # Create results directory if it doesn't exist
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # Run experiment
    run_experiment(config)

if __name__ == "__main__":
    main()