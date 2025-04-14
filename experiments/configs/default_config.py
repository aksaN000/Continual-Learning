"""
Default configuration for continual learning experiments.
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
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
    },
    
    # Continual learning settings
    'continual': {
        'replay_buffer_size': 1000,  # Store more examples
        'replay_batch_size': 16,     # Use more examples per batch
        'ewc_lambda': 100.0,         # Much stronger regularization
    },
    
    # Experiment settings
    'experiment': {
        'name': 'default',
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