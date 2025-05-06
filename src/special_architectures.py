#!/usr/bin/env python3

"""
Special handlers for architecture-specific training and evaluation routines.
This file provides custom implementations for architectures that require
special handling like Siamese networks and ArcFace.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .face_models import get_model, get_criterion
from .base_config import logger
from .training_utils import apply_gradient_clipping, save_checkpoint
from .advanced_metrics import create_enhanced_confusion_matrix, calculate_per_class_metrics, expected_calibration_error

class SiameseDataset(Dataset):
    """Dataset for Siamese networks that generates pairs of images."""
    
    def __init__(self, base_dataset, same_pair_ratio=0.5):
        """
        Initialize the Siamese dataset wrapper.
        
        Args:
            base_dataset: The original dataset (e.g., ImageFolder)
            same_pair_ratio: Ratio of same-class pairs to generate
        """
        self.base_dataset = base_dataset
        self.same_pair_ratio = same_pair_ratio
        
        # Group samples by class
        self.class_indices = {}
        for idx, (_, label) in enumerate(self.base_dataset.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get the first image and its label
        img1, label1 = self.base_dataset[idx]
        
        # Decide whether to create a same-class pair or different-class pair
        if torch.rand(1).item() < self.same_pair_ratio:
            # Same class pair (generate 1 for same class)
            target = 0
            # Get another image from the same class
            if len(self.class_indices[label1]) > 1:
                idx2 = np.random.choice([i for i in self.class_indices[label1] if i != idx])
                img2, _ = self.base_dataset[idx2]
            else:
                # If there's only one image in this class, use the same image
                img2 = img1
        else:
            # Different class pair (generate 0 for different class)
            target = 1
            # Get an image from a different class
            other_classes = [c for c in self.class_indices.keys() if c != label1]
            if other_classes:
                other_class = np.random.choice(other_classes)
                idx2 = np.random.choice(self.class_indices[other_class])
                img2, _ = self.base_dataset[idx2]
            else:
                # Fall back to same class if no other classes exist
                target = 0
                img2 = img1
        
        return (img1, img2), target

def train_siamese_network(model, train_dataset, val_dataset, config, results_manager):
    """
    Custom training routine for Siamese networks.
    
    Args:
        model: The Siamese network model
        train_dataset: Original training dataset
        val_dataset: Original validation dataset 
        config: Experiment configuration
        results_manager: Results manager for logging metrics
        
    Returns:
        Dict: Training summary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create Siamese datasets
    siamese_train_dataset = SiameseDataset(train_dataset)
    siamese_val_dataset = SiameseDataset(val_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        siamese_train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        siamese_val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Get Siamese criterion (contrastive loss)
    criterion = get_criterion('siamese')
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Set up learning rate scheduler if requested
    scheduler = None
    if hasattr(config, 'lr_scheduler_type') and config.lr_scheduler_type != 'none':
        from .training_utils import get_scheduler
        scheduler_type = config.lr_scheduler_type
        scheduler_params = config.lr_scheduler_params or {}
        
        # Calculate steps_per_epoch if needed
        if scheduler_type == "one_cycle":
            scheduler_params['steps_per_epoch'] = len(train_loader)
            scheduler_params['epochs'] = config.epochs
        
        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            **scheduler_params
        )
    
    # Set up early stopping if requested
    early_stopping = None
    if hasattr(config, 'use_early_stopping') and config.use_early_stopping:
        from .training_utils import EarlyStopping
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=config.early_stopping_mode
        )
    
    # Training loop
    best_val_accuracy = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    start_epoch = 1
    
    # Check if training should be resumed from checkpoint
    if hasattr(config, 'resumable_training') and config.resumable_training:
        checkpoints_dir = Path(config.results_dir) / "checkpoints"
        if checkpoints_dir.exists():
            # Fix: correctly use list() on the result of glob()
            checkpoint_files = list((Path(config.results_dir) / "checkpoints").glob("checkpoint_epoch_*.pth"))
            if checkpoint_files:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
                latest_checkpoint = checkpoint_files[-1]
                
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore scheduler state if available
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Set start epoch
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Starting from epoch {start_epoch}")
    
    import time
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        
        epoch_start_time = time.time()
        
        for (img1, img2), labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping if enabled
            if hasattr(config, 'use_gradient_clipping') and config.use_gradient_clipping:
                apply_gradient_clipping(
                    model=model,
                    max_norm=config.gradient_clipping_max_norm,
                    adaptive=config.gradient_clipping_adaptive,
                    model_type='siamese'
                )
            
            # Optimize
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Record training metrics
        results_manager.record_training_metrics(epoch, {
            "loss": epoch_loss,
            "epoch_time": time.time() - epoch_start_time
        })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for (img1, img2), labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
                
                # Forward pass
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)
                
                # Calculate accuracy (0 if distance < 0.5, 1 otherwise)
                distances = F.pairwise_distance(output1, output2)
                predictions = (distances > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Record validation metrics
        val_metrics = {
            "loss": epoch_val_loss,
            "accuracy": accuracy
        }
        results_manager.record_evaluation_metrics(epoch, val_metrics)
        
        # Step learning rate scheduler if it's a validation-based scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_value = epoch_val_loss if hasattr(config, 'early_stopping_metric') and config.early_stopping_metric == "loss" else -accuracy
                scheduler.step(metric_value)
        
        # Save checkpoint if this is the best model
        is_best = False
        if hasattr(config, 'early_stopping_mode'):
            if config.early_stopping_mode == 'max':
                # For metrics like accuracy where higher is better
                is_best = accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = accuracy
            else:
                # For metrics like loss where lower is better
                is_best = epoch_val_loss < best_val_loss
                if is_best:
                    best_val_loss = epoch_val_loss
        else:
            # Default to accuracy if not specified
            is_best = accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = accuracy
        
        # Save checkpoint with appropriate frequency
        if hasattr(config, 'checkpoint_frequency') and (epoch % config.checkpoint_frequency == 0 or is_best):
            results_manager.save_model_checkpoint(
                model=model, 
                optimizer=optimizer, 
                epoch=epoch, 
                is_best=is_best,
                scheduler=scheduler,
                metrics=val_metrics
            )
        
        # Log progress
        logger.info(f'Epoch {epoch}/{config.epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time: {time.time() - epoch_start_time:.2f}s')
        
        # Step learning rate scheduler if it's an epoch-based scheduler
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Check early stopping if enabled
        if early_stopping:
            # Use appropriate metric for early stopping
            if hasattr(config, 'early_stopping_metric') and config.early_stopping_metric == "loss":
                improvement = early_stopping(epoch_val_loss)
            else:  # Use accuracy
                # Negate accuracy for 'min' mode
                es_value = -accuracy if early_stopping.mode == 'min' else accuracy
                improvement = early_stopping(es_value)
            
            # Check if training should be stopped
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                # Save early stopping trace to a file
                with open(Path(config.results_dir) / "logs" / "early_stopping_trace.json", 'w') as f:
                    import json
                    json.dump({
                        "trace": early_stopping.trace,
                        "stopped_epoch": epoch,
                        "best_score": early_stopping.best_score,
                        "mode": early_stopping.mode
                    }, f, indent=2)
                break
    
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time:.2f}s")
    
    # Record learning curves
    results_manager.record_learning_curves(train_losses, val_losses, val_accuracies)
    
    # Test the best model
    logger.info("Testing the best model...")
    model.load_state_dict(torch.load(Path(config.results_dir) / "checkpoints" / "best_model.pth"))
    
    # Here we'd run evaluation on the test set, but for now return the results
    return {
        "training_time": total_training_time,
        "epochs": epoch - start_epoch + 1,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_loss": best_val_loss,
        "test_metrics": [{
            "accuracy": best_val_accuracy,
            "loss": best_val_loss
        }]
    }

def evaluate_siamese_network(model, test_dataset, config, results_manager):
    """
    Evaluate a Siamese network on test data.
    
    Args:
        model: The Siamese network model
        test_dataset: Original test dataset
        config: Experiment configuration
        results_manager: Results manager for logging metrics
        
    Returns:
        Dict: Evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create Siamese dataset
    siamese_test_dataset = SiameseDataset(test_dataset)
    
    # Create data loader
    test_loader = DataLoader(
        siamese_test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Get Siamese criterion (contrastive loss)
    criterion = get_criterion('siamese')
    
    # Evaluation metrics
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For detailed metrics
    all_distances = []
    all_labels = []
    all_preds = []
    
    import time
    inference_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, ((img1, img2), labels) in enumerate(test_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            # Calculate distances
            distances = F.pairwise_distance(output1, output2)
            predictions = (distances > 0.5).float()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            test_loss += loss.item()
            
            # Store predictions for metrics calculation
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    
    inference_time = time.time() - inference_start_time
    
    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    
    # Convert to numpy arrays for metric calculations
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate precision, recall, F1 score
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    # Create test metrics dictionary
    test_metrics = {
        "accuracy": accuracy,
        "loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_time_total": inference_time,
        "inference_time_per_sample": inference_time / total,
        "inference_samples_per_second": total / inference_time
    }
    
    # Record test metrics
    results_manager.record_test_metrics(test_metrics)
    
    # Record additional metrics if enhanced evaluation is enabled
    if hasattr(config, 'evaluation_mode') and config.evaluation_mode == 'enhanced':
        # Create binary class names for confusion matrix
        binary_classes = ["Same", "Different"]
        
        # Create confusion matrix
        results_manager.record_confusion_matrix(all_labels, all_preds, binary_classes)
        
        # Create calibration metrics
        results_manager.record_calibration_metrics(all_labels, all_preds, all_distances.reshape(-1, 1))
        
        # Create per-class metrics
        results_manager.record_per_class_metrics(all_labels, all_preds, all_distances.reshape(-1, 1), binary_classes)
        
    return test_metrics

class ArcFaceTrainer:
    """
    Specialized trainer for ArcFace networks.
    """
    
    @staticmethod
    def train_arcface_network(model, train_dataset, val_dataset, test_dataset, config, results_manager):
        """
        Custom training routine for ArcFace networks.
        
        Args:
            model: The ArcFace network model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            config: Experiment configuration
            results_manager: Results manager for logging metrics
            
        Returns:
            Dict: Training and testing results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Get criterion
        criterion = get_criterion('arcface')
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Set up learning rate scheduler if requested
        scheduler = None
        if hasattr(config, 'lr_scheduler_type') and config.lr_scheduler_type != 'none':
            from .training_utils import get_scheduler
            scheduler_type = config.lr_scheduler_type
            scheduler_params = config.lr_scheduler_params or {}
            
            scheduler = get_scheduler(
                scheduler_type=scheduler_type,
                optimizer=optimizer,
                **scheduler_params
            )
        
        # Set up early stopping if requested
        early_stopping = None
        if hasattr(config, 'use_early_stopping') and config.use_early_stopping:
            from .training_utils import EarlyStopping
            early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode=config.early_stopping_mode
            )
        
        # Training loop
        best_val_accuracy = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        start_epoch = 1
        
        # Check if training should be resumed from checkpoint
        if hasattr(config, 'resumable_training') and config.resumable_training:
            checkpoints_dir = Path(config.results_dir) / "checkpoints"
            if checkpoints_dir.exists():
                # Fix: correctly use list() on the result of glob()
                checkpoint_files = list((Path(config.results_dir) / "checkpoints").glob("checkpoint_epoch_*.pth"))
                if checkpoint_files:
                    # Sort by epoch number
                    checkpoint_files.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
                    latest_checkpoint = checkpoint_files[-1]
                    
                    logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Restore scheduler state if available
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # Set start epoch
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Starting from epoch {start_epoch}")
        
        import time
        training_start_time = time.time()
        
        for epoch in range(start_epoch, config.epochs + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            
            epoch_start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - ArcFace requires labels during training
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if enabled
                if hasattr(config, 'use_gradient_clipping') and config.use_gradient_clipping:
                    apply_gradient_clipping(
                        model=model,
                        max_norm=config.gradient_clipping_max_norm,
                        adaptive=config.gradient_clipping_adaptive,
                        model_type='arcface'
                    )
                
                # Optimize
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Record training metrics
            results_manager.record_training_metrics(epoch, {
                "loss": epoch_loss,
                "epoch_time": time.time() - epoch_start_time
            })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            val_embeddings = []
            val_labels_list = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Get embeddings for evaluation
                    embeddings = model(inputs)
                    val_embeddings.append(embeddings)
                    val_labels_list.append(labels)
                    
                    # Calculate similarity for classification
                    logits = F.linear(
                        F.normalize(embeddings), 
                        F.normalize(model.arcface.weight)
                    )
                    loss = criterion(logits, labels)
                    
                    _, preds = torch.max(logits, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    val_loss += loss.item()
            
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            
            # Record validation metrics
            val_metrics = {
                "loss": epoch_val_loss,
                "accuracy": accuracy
            }
            results_manager.record_evaluation_metrics(epoch, val_metrics)
            
            # Step learning rate scheduler if it's a validation-based scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_value = epoch_val_loss if hasattr(config, 'early_stopping_metric') and config.early_stopping_metric == "loss" else -accuracy
                    scheduler.step(metric_value)
            
            # Save checkpoint if this is the best model
            is_best = False
            if hasattr(config, 'early_stopping_mode'):
                if config.early_stopping_mode == 'max':
                    # For metrics like accuracy where higher is better
                    is_best = accuracy > best_val_accuracy
                    if is_best:
                        best_val_accuracy = accuracy
                else:
                    # For metrics like loss where lower is better
                    is_best = epoch_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = epoch_val_loss
            else:
                # Default to accuracy if not specified
                is_best = accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = accuracy
            
            # Save checkpoint with appropriate frequency
            if hasattr(config, 'checkpoint_frequency') and (epoch % config.checkpoint_frequency == 0 or is_best):
                results_manager.save_model_checkpoint(
                    model=model, 
                    optimizer=optimizer, 
                    epoch=epoch, 
                    is_best=is_best,
                    scheduler=scheduler,
                    metrics=val_metrics
                )
            
            # Log progress
            logger.info(f'Epoch {epoch}/{config.epochs}, '
                      f'Train Loss: {epoch_loss:.4f}, '
                      f'Val Loss: {epoch_val_loss:.4f}, '
                      f'Accuracy: {accuracy:.2f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                      f'Time: {time.time() - epoch_start_time:.2f}s')
            
            # Step learning rate scheduler if it's an epoch-based scheduler
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Check early stopping if enabled
            if early_stopping:
                # Use appropriate metric for early stopping
                if hasattr(config, 'early_stopping_metric') and config.early_stopping_metric == "loss":
                    improvement = early_stopping(epoch_val_loss)
                else:  # Use accuracy
                    # Negate accuracy for 'min' mode
                    es_value = -accuracy if early_stopping.mode == 'min' else accuracy
                    improvement = early_stopping(es_value)
                
                # Check if training should be stopped
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    # Save early stopping trace to a file
                    with open(Path(config.results_dir) / "logs" / "early_stopping_trace.json", 'w') as f:
                        import json
                        json.dump({
                            "trace": early_stopping.trace,
                            "stopped_epoch": epoch,
                            "best_score": early_stopping.best_score,
                            "mode": early_stopping.mode
                        }, f, indent=2)
                    break
        
        total_training_time = time.time() - training_start_time
        logger.info(f"Training completed in {total_training_time:.2f}s")
        
        # Record learning curves
        results_manager.record_learning_curves(train_losses, val_losses, val_accuracies)
        
        # Test the best model
        logger.info("Testing the best model...")
        model.load_state_dict(torch.load(Path(config.results_dir) / "checkpoints" / "best_model.pth"))
        model.eval()
        
        # Perform evaluation on test set
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_embeddings = []
        all_labels = []
        all_preds = []
        all_probs = []
        
        inference_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                batch_start_time = time.time()
                
                # Get embeddings for evaluation
                embeddings = model(inputs)
                
                # Calculate similarity for classification
                logits = F.linear(
                    F.normalize(embeddings), 
                    F.normalize(model.arcface.weight)
                )
                
                # Calculate softmax probabilities
                probs = F.softmax(logits, dim=1)
                
                loss = criterion(logits, labels)
                
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                test_loss += loss.item()
                
                # Store predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Log batch time for first batch (typically includes warm-up overhead)
                if batch_idx == 0:
                    logger.info(f"Batch {batch_idx} completed in {time.time() - batch_start_time:.2f}s")
        
        inference_time = time.time() - inference_start_time
        
        # Calculate metrics
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        # Calculate precision, recall, F1 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Create test metrics dictionary
        test_metrics = {
            "accuracy": accuracy,
            "loss": test_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "inference_time_total": inference_time,
            "inference_time_per_sample": inference_time / total,
            "inference_samples_per_second": total / inference_time
        }
        
        # Record test metrics
        results_manager.record_test_metrics(test_metrics)
        
        # Record additional metrics if enhanced evaluation is enabled
        if hasattr(config, 'evaluation_mode') and config.evaluation_mode == 'enhanced':
            # Get class names
            class_names = test_dataset.classes
            
            # Record confusion matrix
            results_manager.record_confusion_matrix(all_labels, all_preds, class_names)
            
            # Record per-class metrics
            results_manager.record_per_class_metrics(all_labels, all_preds, all_probs, class_names)
            
            # Record calibration metrics
            results_manager.record_calibration_metrics(all_labels, all_preds, all_probs)
            
            # Save raw predictions
            results_manager.save_raw_predictions(all_labels, all_preds, all_probs, class_names)
        
        # Return training summary
        return {
            "training_time": total_training_time,
            "epochs": epoch - start_epoch + 1,
            "best_validation_accuracy": best_val_accuracy,
            "best_validation_loss": best_val_loss,
            "test_metrics": [test_metrics]
        }


def handle_special_architecture(architecture, model, train_dataset, val_dataset, test_dataset, config, results_manager):
    """
    Handle special architectures with custom training routines.
    
    Args:
        architecture: Architecture name
        model: The model instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Experiment configuration
        results_manager: Results manager for logging metrics
        
    Returns:
        Dict: Results from training and testing
        bool: Whether this architecture was handled specially
    """
    if architecture == 'siamese':
        # Handle Siamese network
        training_results = train_siamese_network(model, train_dataset, val_dataset, config, results_manager)
        
        # Evaluate on test set
        test_metrics = evaluate_siamese_network(model, test_dataset, config, results_manager)
        
        # Add test metrics to results
        if 'test_metrics' not in training_results:
            training_results['test_metrics'] = []
        training_results['test_metrics'].append(test_metrics)
        
        return training_results, True
        
    elif architecture == 'arcface':
        # Handle ArcFace network
        results = ArcFaceTrainer.train_arcface_network(
            model, train_dataset, val_dataset, test_dataset, config, results_manager
        )
        return results, True
        
    # Not a special architecture
    return None, False 