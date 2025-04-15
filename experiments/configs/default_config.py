"""
Optimized configuration for combined EWC+Replay continual learning.
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
        'learning_rate': 1e-5,  # Lower learning rate for better stability
        'weight_decay': 0.01,
    },
    
    # Continual learning settings
    'continual': {
        'replay_buffer_size': 750,      # Larger buffer size
        'replay_batch_size': 10,        # Larger batch size for replay
        'ewc_lambda': 10.0,             # Stronger EWC regularization
        'replay_strategy': 'balanced',  # balanced, importance, or diversity
    },
    
    # Experiment settings
    'experiment': {
        'name': 'optimized_combined',
        'seed': 42,
        'use_ewc': True,
        'use_replay': True,
        'online_ewc': True,           # Use online EWC (more efficient)
        'importance_threshold': 85,    # Percentile threshold for parameter importance
    },
    
    # Output settings
    'output': {
        'save_model': True,
        'save_dir': 'results',
        'log_interval': 10,
    }
}