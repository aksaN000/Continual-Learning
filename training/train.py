import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from training.replay_buffer import ReplayBuffer
from training.metrics import ContinualMetrics

def train_on_domain(model, domain_idx, train_loader, val_loader, 
                   replay_buffer, device, epochs=3, 
                   replay_batch_size=8, learning_rate=5e-5):
    """
    Train the model on a specific domain with experience replay.
    
    Args:
        model: ContinualTextCommandLearner model
        domain_idx: Index of the current domain
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        replay_buffer: Replay buffer for experience replay
        device: Device to run training on
        epochs: Number of epochs to train
        replay_batch_size: Batch size for replay samples
        learning_rate: Learning rate for optimizer
        
    Returns:
        val_acc: Validation accuracy on this domain
    """
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_ewc_loss = 0
        total_replay_loss = 0
        total_ce_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get domain-specific logits
            domain_logits = model.get_domain_logits(outputs, domain_idx)
            
            # Compute cross-entropy loss
            ce_loss = criterion(domain_logits, labels)
            total_ce_loss += ce_loss.item()
            
            # Initialize total loss with CE loss
            loss = ce_loss
            
            # Add EWC regularization if enabled (model.ewc_lambda > 0)
            ewc_loss = model.compute_ewc_loss()
            ewc_loss_value = ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss
            
            if ewc_loss_value > 0:
                loss += ewc_loss
                total_ewc_loss += ewc_loss_value
            
            # Initialize replay loss value
            replay_loss_value = 0
            
            # Sample from replay buffer if it's not empty and we're not on the first domain
            # (no previous data to replay for the first domain)
            if len(replay_buffer) > 0 and replay_batch_size > 0 and domain_idx > 0:
                # Sample examples from the replay buffer with a focus on previous domains
                replay_batch = replay_buffer.sample(replay_batch_size, device)
                
                # Forward pass for replay batch
                replay_outputs = model(replay_batch['input_ids'], replay_batch['attention_mask'])
                
                # Compute loss for replay examples
                # Use a lower weight for replay loss to balance with current domain learning
                replay_loss = 0.5 * criterion(replay_outputs, replay_batch['label'])
                replay_loss_value = replay_loss.item()
                total_replay_loss += replay_loss_value
                
                # Add replay loss to total loss
                loss += replay_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(domain_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            train_acc = 100 * correct / total
            
            # Update progress bar with detailed loss breakdown
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{train_acc:.2f}%",
                'ewc': f"{ewc_loss_value:.4f}",
                'replay': f"{replay_loss_value:.4f}"
            })
            
            # Add current batch to replay buffer (do this for all domains)
            # For better memory management, only add a subset of each batch
            if domain_idx < model.num_domains - 1:  # Don't add examples from last domain
                # Add with probability based on accuracy - prioritize correct examples
                sample_prob = 0.3  # Base probability to add examples
                replay_buffer.add_batch(batch, device, sample_prob=sample_prob)
        
        # Print epoch results with loss breakdown
        avg_total_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_ewc_loss = total_ewc_loss / len(train_loader)
        avg_replay_loss = total_replay_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_total_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  CE Loss: {avg_ce_loss:.4f}, EWC Loss: {avg_ewc_loss:.4f}, Replay Loss: {avg_replay_loss:.4f}")
    
    # Validate on this domain
    val_acc = evaluate_on_domain(model, domain_idx, val_loader, device)
    print(f"Validation accuracy on domain {domain_idx}: {val_acc:.4f}")
    
    # Update EWC parameters for regularization in future domains
    # Only update if EWC is enabled (model.ewc_lambda > 0) and we're not on the last domain
    if domain_idx < model.num_domains - 1:  # No need to update for the last domain
        model.update_ewc_params(train_loader, device)
    
    return val_acc

def evaluate_on_domain(model, domain_idx, data_loader, device):
    """
    Evaluate the model on a specific domain.
    
    Args:
        model: ContinualTextCommandLearner model
        domain_idx: Index of the domain to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        
    Returns:
        accuracy: Accuracy on this domain
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get domain-specific logits
            domain_logits = model.get_domain_logits(outputs, domain_idx)
            
            # Get predictions
            _, preds = torch.max(domain_logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return accuracy

def evaluate_all_domains(model, domain_loaders, device, metrics=None):
    """
    Evaluate the model on all domains seen so far.
    
    Args:
        model: ContinualTextCommandLearner model
        domain_loaders: List of DataLoaders for each domain
        device: Device to run evaluation on
        metrics: ContinualMetrics object (optional)
        
    Returns:
        results: List of accuracies for each domain
    """
    results = []
    
    for domain_idx, loader in enumerate(domain_loaders):
        if loader is not None:
            acc = evaluate_on_domain(model, domain_idx, loader, device)
            results.append(acc)
            print(f"Domain {domain_idx} accuracy: {acc:.4f}")
            
            # Update metrics if provided
            if metrics is not None:
                # For simplicity, we're not calculating true and pred here
                # In a real implementation, you would collect these during evaluation
                # For now, we'll just pass the accuracy directly
                y_true = np.zeros(1)  # Placeholder
                y_pred = np.zeros(1)  # Placeholder
                metrics.domain_accs[domain_idx].append(acc)
    
    # Calculate average accuracy
    if results:
        avg_acc = np.mean(results)
        print(f"Average accuracy across all domains: {avg_acc:.4f}")
    
    return results

def train_continual_learning(model, domains, data_loaders, replay_buffer, device, 
                            epochs=3, replay_batch_size=8, learning_rate=5e-5):
    """
    Train the model sequentially on multiple domains.
    
    Args:
        model: ContinualTextCommandLearner model
        domains: List of domain names
        data_loaders: Dictionary mapping domain names to (train_loader, val_loader) tuples
        replay_buffer: Replay buffer for experience replay
        device: Device to run training on
        epochs: Number of epochs to train on each domain
        replay_batch_size: Batch size for replay samples
        learning_rate: Learning rate for optimizer
        
    Returns:
        metrics: ContinualMetrics object with performance tracking
    """
    # Initialize metrics tracker
    metrics = ContinualMetrics(len(domains))
    
    # Store validation loaders for evaluation
    val_loaders = []
    
    # Train on each domain sequentially
    for domain_idx, domain_name in enumerate(domains):
        print(f"\n{'='*50}")
        print(f"Training on domain {domain_idx}: {domain_name}")
        print(f"{'='*50}")
        
        # Get data loaders for this domain
        train_loader, val_loader = data_loaders[domain_name]
        val_loaders.append(val_loader)
        
        # Train on this domain
        train_on_domain(
            model, 
            domain_idx, 
            train_loader,
            val_loader,
            replay_buffer,
            device,
            epochs=epochs,
            replay_batch_size=replay_batch_size,
            learning_rate=learning_rate
        )
        
        # Evaluate on all domains seen so far
        print("\nEvaluating on all domains seen so far:")
        evaluate_all_domains(model, val_loaders, device, metrics)
        
        # Log metrics
        metrics.log_metrics(domain_idx)
    
    # Final evaluation
    print("\nFinal evaluation on all domains:")
    evaluate_all_domains(model, val_loaders, device, metrics)
    
    return metrics