"""
Improved configuration for continual learning experiments.
"""

config = {
    # Data settings
    'data': {
        'data_dir': 'data/processed',
        'max_length': 128,
        'batch_size': 16,
    },
    
    # Model settings
    'model': {
        'base_model': 'bert-base-uncased',
        'dropout': 0.1,
    },
    
    # Training settings
    'training': {
        'epochs': 3,
        'learning_rate': 2e-5,  # Slightly lower learning rate
        'weight_decay': 0.01,
    },
    
    # Continual learning settings
    'continual': {
        'replay_buffer_size': 500,  # More reasonable buffer size
        'replay_batch_size': 8,     # Smaller batch size to maintain balance
        'ewc_lambda': 5.0,          # Much lower regularization strength
    },
    
    # Experiment settings
    'experiment': {
        'name': 'improved_config',
        'seed': 42,
        'use_ewc': True,
        'use_replay': True,
    },
    
    # Output settings
    'output': {
        'save_model': True,
        'save_dir': 'results',
        'log_interval': 10,
    }
}