#!/usr/bin/env python3

import os
import sys
import json
import uuid
import time
import logging
import datetime
import shutil
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import optuna
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import torch.nn.functional as F

from .base_config import PROJECT_ROOT, CHECKPOINTS_DIR, OUT_DIR, logger
from .data_prep import PreprocessingConfig, process_raw_data
from .face_models import get_model, get_criterion
from .testing import evaluate_model, plot_confusion_matrix, plot_roc_curves, plot_gradcam_visualization
from .visualize import plot_tsne_embeddings, plot_attention_maps, plot_embedding_similarity, plot_learning_curves


class ExperimentConfig:
    """Configuration for face recognition experiments."""
    
    class Dataset(Enum):
        DATASET1 = "dataset1"  # High diversity of subjects (36 people, 49 images each)
        DATASET2 = "dataset2"  # High quantity per person (18 people, 100 images each)
        BOTH = "both"          # Both datasets
    
    class ModelArchitecture(Enum):
        BASELINE = "baseline"
        CNN = "cnn"  # ResNet transfer learning
        SIAMESE = "siamese"
        ATTENTION = "attention"
        ARCFACE = "arcface"
        HYBRID = "hybrid"
    
    def __init__(self,
                 experiment_id: str = None,
                 experiment_name: str = "Unnamed Experiment",
                 dataset: Union[Dataset, str] = Dataset.BOTH,
                 model_architecture: Union[ModelArchitecture, str] = ModelArchitecture.CNN,
                 preprocessing_config: PreprocessingConfig = None,
                 epochs: int = 30,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 cross_dataset_testing: bool = True,
                 results_dir: str = None):
        """Initialize experiment configuration."""
        # Generate UUID if not provided
        self.experiment_id = experiment_id or str(uuid.uuid4())[:8]
        self.experiment_name = experiment_name
        
        # Convert string to enum if needed
        if isinstance(dataset, str):
            self.dataset = self.Dataset(dataset)
        else:
            self.dataset = dataset
            
        if isinstance(model_architecture, str):
            self.model_architecture = self.ModelArchitecture(model_architecture)
        else:
            self.model_architecture = model_architecture
        
        # Set preprocessing configuration
        self.preprocessing_config = preprocessing_config
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Testing configuration
        self.cross_dataset_testing = cross_dataset_testing
        
        # Results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = OUT_DIR / f"experiment_{self.experiment_id}_{timestamp}"

        # Creation timestamp
        self.created_at = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        config_dict = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "dataset": self.dataset.value,
            "model_architecture": self.model_architecture.value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "cross_dataset_testing": self.cross_dataset_testing,
            "results_dir": str(self.results_dir),
            "created_at": self.created_at
        }
        
        # Add preprocessing config if available
        if self.preprocessing_config:
            config_dict["preprocessing_config"] = self.preprocessing_config.to_dict()
            
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Extract preprocessing config if available
        preprocessing_config = None
        if "preprocessing_config" in config_dict:
            preprocessing_config = PreprocessingConfig.from_dict(config_dict["preprocessing_config"])
            
        # Create the basic config
        config = cls(
            experiment_id=config_dict.get("experiment_id"),
            experiment_name=config_dict.get("experiment_name", "Unnamed Experiment"),
            dataset=config_dict.get("dataset", "both"),
            model_architecture=config_dict.get("model_architecture", "cnn"),
            preprocessing_config=preprocessing_config,
            epochs=config_dict.get("epochs", 30),
            batch_size=config_dict.get("batch_size", 32),
            learning_rate=config_dict.get("learning_rate", 0.001),
            cross_dataset_testing=config_dict.get("cross_dataset_testing", True),
            results_dir=config_dict.get("results_dir")
        )
        
        # Set creation timestamp if available
        if "created_at" in config_dict:
            config.created_at = config_dict["created_at"]
            
        return config
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save config to JSON file."""
        if filepath is None:
            filepath = self.results_dir / "experiment_config.json"
            
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)


class ModelRegistry:
    """Registry to manage face recognition models."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize model registry."""
        self.registry_path = registry_path or CHECKPOINTS_DIR / "model_registry.json"
        self.models = {}
        
        # Load existing registry if it exists
        if self.registry_path.exists():
            self.load()
    
    def register_model(self, 
                      model_name: str, 
                      architecture: str,
                      dataset_name: str,
                      experiment_id: str,
                      parameters: Dict[str, Any],
                      metrics: Optional[Dict[str, float]] = None,
                      checkpoint_path: Optional[Path] = None) -> str:
        """Register a new model with metadata."""
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if model name exists, add versioning if needed
        base_name = model_name
        version = 1
        
        while model_name in self.models:
            model_name = f"{base_name}_v{version}"
            version += 1
        
        # Create model metadata
        model_metadata = {
            "model_name": model_name,
            "architecture": architecture,
            "dataset": dataset_name,
            "experiment_id": experiment_id,
            "parameters": parameters,
            "created_at": timestamp,
            "metrics": metrics or {},
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None
        }
        
        # Add to registry
        self.models[model_name] = model_metadata
        
        # Save registry
        self.save()
        
        return model_name
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata by name."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return self.models[model_name]
    
    def list_models(self, 
                   architecture: Optional[str] = None, 
                   dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models with optional filtering."""
        filtered_models = self.models.values()
        
        if architecture:
            filtered_models = [m for m in filtered_models if m["architecture"] == architecture]
            
        if dataset:
            filtered_models = [m for m in filtered_models if m["dataset"] == dataset]
            
        return list(filtered_models)
    
    def update_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Update model metrics."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        self.models[model_name]["metrics"].update(metrics)
        self.models[model_name]["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save registry
        self.save()
    
    def save(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def load(self) -> None:
        """Load registry from disk."""
        if not self.registry_path.exists():
            self.models = {}
            return
        
        with open(self.registry_path, 'r') as f:
            self.models = json.load(f)


class ResultsManager:
    """Manager for experiment results and metrics."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        """Initialize results manager with experiment configuration."""
        self.config = experiment_config
        self.results_dir = experiment_config.results_dir
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize metrics storage
        self.train_metrics = []
        self.eval_metrics = []
        self.test_metrics = []
        self.confusion_matrices = {}
        self.experiment_log = []
        
        # Add experiment start entry
        self.log_event("experiment_started", {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset.value,
            "model_architecture": self.config.model_architecture.value,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _setup_directories(self):
        """Create directory structure for results."""
        # Main directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.results_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def record_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Record per-epoch training metrics."""
        metrics = {**metrics, "epoch": epoch}
        self.train_metrics.append(metrics)
        
        # Log event
        self.log_event("training_epoch_completed", {
            "epoch": epoch,
            "metrics": metrics
        })
        
        # Save metrics to CSV
        self._save_metrics_to_csv(self.train_metrics, "training_metrics.csv")
    
    def record_evaluation_metrics(self, epoch: int, metrics: Dict[str, float], dataset: str = "validation"):
        """Record evaluation metrics."""
        metrics = {**metrics, "epoch": epoch, "dataset": dataset}
        self.eval_metrics.append(metrics)
        
        # Log event
        self.log_event("evaluation_completed", {
            "epoch": epoch,
            "dataset": dataset,
            "metrics": metrics
        })
        
        # Save metrics to CSV
        self._save_metrics_to_csv(self.eval_metrics, "evaluation_metrics.csv")
    
    def record_test_metrics(self, metrics: Dict[str, float], dataset: str = "test"):
        """Record test metrics after training is complete."""
        metrics = {**metrics, "dataset": dataset}
        self.test_metrics.append(metrics)
        
        # Log event
        self.log_event("testing_completed", {
            "dataset": dataset,
            "metrics": metrics
        })
        
        # Save metrics to CSV
        self._save_metrics_to_csv(self.test_metrics, "test_metrics.csv")
    
    def record_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                              class_names: List[str], dataset: str = "test"):
        """Record confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Store confusion matrix
        self.confusion_matrices[dataset] = {
            "matrix": cm.tolist(),
            "class_names": class_names
        }
        
        # Save to JSON
        cm_path = self.metrics_dir / f"confusion_matrix_{dataset}.json"
        with open(cm_path, 'w') as f:
            json.dump(self.confusion_matrices[dataset], f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {dataset}')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.plots_dir / f"confusion_matrix_{dataset}.png")
        plt.close()
    
    def record_learning_curves(self, train_losses: List[float], val_losses: List[float], 
                             accuracies: List[float]):
        """Record and visualize learning curves."""
        # Save data
        curves_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "accuracies": accuracies
        }
        
        curves_path = self.metrics_dir / "learning_curves.json"
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f, indent=2)
        
        # Create visualization
        plot_learning_curves(train_losses, val_losses, accuracies, 
                          str(self.plots_dir), self.config.experiment_id)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an experiment event with timestamp."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        self.experiment_log.append(log_entry)
        
        # Save log to file
        self._save_experiment_log()
    
    def _save_experiment_log(self):
        """Save experiment log to JSON file."""
        log_path = self.logs_dir / "experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
    
    def _save_metrics_to_csv(self, metrics_list: List[Dict[str, Any]], filename: str):
        """Save metrics to CSV file."""
        if not metrics_list:
            return
        
        df = pd.DataFrame(metrics_list)
        csv_path = self.metrics_dir / filename
        df.to_csv(csv_path, index=False)
    
    def save_model_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                            epoch: int, is_best: bool = False) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pth"
        best_model_path = self.checkpoints_dir / "best_model.pth"
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment_id": self.config.experiment_id
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            torch.save(model.state_dict(), best_model_path)
            self.log_event("best_model_saved", {"epoch": epoch})
        
        # Log event
        self.log_event("checkpoint_saved", {
            "epoch": epoch,
            "path": str(checkpoint_path),
            "is_best": is_best
        })
        
        return checkpoint_path if not is_best else best_model_path
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive experiment summary."""
        # Gather key metrics
        best_val_metrics = {}
        if self.eval_metrics:
            # Find epoch with best validation accuracy
            best_epoch_idx = max(range(len(self.eval_metrics)), 
                               key=lambda i: self.eval_metrics[i].get('accuracy', 0))
            best_val_metrics = self.eval_metrics[best_epoch_idx]
        
        # Compute average test metrics across datasets
        avg_test_metrics = {}
        if self.test_metrics:
            metrics_keys = set()
            for metrics in self.test_metrics:
                metrics_keys.update(metrics.keys())
            
            metrics_keys.discard('dataset')  # Remove non-numeric field
            
            # Calculate averages
            for key in metrics_keys:
                values = [m.get(key, 0) for m in self.test_metrics if key in m]
                if values:
                    avg_test_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        # Create summary
        summary = {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset.value,
            "model_architecture": self.config.model_architecture.value,
            "preprocessing": self.config.preprocessing_config.name if self.config.preprocessing_config else "None",
            "training_params": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate
            },
            "best_validation_metrics": best_val_metrics,
            "test_metrics": self.test_metrics,
            "average_test_metrics": avg_test_metrics,
            "completed_at": datetime.datetime.now().isoformat()
        }
        
        # Save summary to file
        summary_path = self.results_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log event
        self.log_event("experiment_completed", {
            "summary": summary
        })
        
        return summary
    
    def export_to_csv(self, filepath: Optional[Path] = None) -> Path:
        """Export all metrics to a single CSV file."""
        if filepath is None:
            filepath = self.results_dir / "all_metrics.csv"
        
        # Combine all metrics
        all_metrics = []
        
        # Add training metrics
        for metric in self.train_metrics:
            metric_copy = metric.copy()
            metric_copy['phase'] = 'training'
            all_metrics.append(metric_copy)
        
        # Add evaluation metrics
        for metric in self.eval_metrics:
            metric_copy = metric.copy()
            metric_copy['phase'] = 'evaluation'
            all_metrics.append(metric_copy)
        
        # Add test metrics
        for metric in self.test_metrics:
            metric_copy = metric.copy()
            metric_copy['phase'] = 'test'
            all_metrics.append(metric_copy)
        
        # Create DataFrame and save
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            df.to_csv(filepath, index=False)
        
        return filepath 


class ExperimentManager:
    """Manager for running face recognition experiments."""
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        """Initialize experiment manager."""
        self.model_registry = model_registry or ModelRegistry()
        
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a face recognition experiment according to configuration."""
        # Set up results manager
        results_manager = ResultsManager(config)
        
        # Log experiment start
        logger.info(f"Starting experiment: {config.experiment_name} (ID: {config.experiment_id})")
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Save experiment configuration
        config.save()
        
        # Determine experiment type based on configuration
        if config.dataset == ExperimentConfig.Dataset.BOTH and config.cross_dataset_testing:
            return self._run_cross_dataset_experiment(config, results_manager)
        elif len(config.model_architecture.value) == 1:
            return self._run_single_model_experiment(config, results_manager)
        else:
            return self._run_architecture_comparison_experiment(config, results_manager)
    
    def _run_single_model_experiment(self, config: ExperimentConfig, 
                                   results_manager: ResultsManager) -> Dict[str, Any]:
        """Run an experiment with a single model architecture on one dataset."""
        try:
            # Set up device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model
            model_arch = config.model_architecture.value
            model = get_model(model_arch)
            model = model.to(device)
            
            # Set up optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            
            # Get criterion based on model type
            criterion = get_criterion(model_arch)
            
            # Setup data loaders
            train_loader, val_loader, test_loader, class_names = self._setup_data_loaders(
                config.dataset.value, 
                config.preprocessing_config, 
                batch_size=config.batch_size
            )
            
            # Training loop
            best_val_accuracy = 0
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            for epoch in range(1, config.epochs + 1):
                # Training phase
                model.train()
                running_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if model_arch == 'siamese':
                        output1, output2 = model(inputs[0], inputs[1])
                        loss = criterion(output1, output2, labels)
                    elif model_arch == 'arcface':
                        outputs = model(inputs, labels)
                        loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                epoch_loss = running_loss / len(train_loader)
                train_losses.append(epoch_loss)
                
                # Record training metrics
                results_manager.record_training_metrics(epoch, {
                    "loss": epoch_loss
                })
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        if model_arch == 'siamese':
                            output1, output2 = model(inputs[0], inputs[1])
                            loss = criterion(output1, output2, labels)
                            preds = (F.pairwise_distance(output1, output2) < 0.5).float()
                            correct += (preds == labels).sum().item()
                        elif model_arch == 'arcface':
                            embeddings = model(inputs)
                            outputs = F.linear(
                                F.normalize(embeddings), 
                                F.normalize(model.arcface.weight)
                            )
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            correct += (preds == labels).sum().item()
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            correct += (preds == labels).sum().item()
                        
                        val_loss += loss.item()
                        total += labels.size(0)
                
                epoch_val_loss = val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)
                
                accuracy = 100 * correct / total
                val_accuracies.append(accuracy)
                
                # Record validation metrics
                results_manager.record_evaluation_metrics(epoch, {
                    "loss": epoch_val_loss,
                    "accuracy": accuracy
                })
                
                # Save checkpoint if this is the best model
                is_best = accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = accuracy
                
                results_manager.save_model_checkpoint(model, optimizer, epoch, is_best)
                
                # Log progress
                logger.info(f'Epoch {epoch}/{config.epochs}, '
                          f'Train Loss: {epoch_loss:.4f}, '
                          f'Val Loss: {epoch_val_loss:.4f}, '
                          f'Accuracy: {accuracy:.2f}%')
            
            # Record learning curves
            results_manager.record_learning_curves(train_losses, val_losses, val_accuracies)
            
            # Test the best model
            logger.info("Testing the best model...")
            model.load_state_dict(torch.load(results_manager.checkpoints_dir / "best_model.pth"))
            model.eval()
            
            # Evaluate on test set
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    if model_arch == 'siamese':
                        output1, output2 = model(inputs[0], inputs[1])
                        preds = (F.pairwise_distance(output1, output2) < 0.5).float()
                        all_probs.extend(F.pairwise_distance(output1, output2).cpu().numpy())
                    elif model_arch == 'arcface':
                        embeddings = model(inputs)
                        outputs = F.linear(
                            F.normalize(embeddings), 
                            F.normalize(model.arcface.weight)
                        )
                        _, preds = torch.max(outputs, 1)
                        all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            test_metrics = {
                "accuracy": accuracy_score(all_labels, all_preds) * 100,
                "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
                "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
                "f1_score": f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            }
            
            # Record test metrics
            results_manager.record_test_metrics(test_metrics)
            
            # Record confusion matrix
            results_manager.record_confusion_matrix(all_labels, all_preds, class_names)
            
            # Register model with registry
            model_name = f"{model_arch}_{config.dataset.value}_{config.experiment_id}"
            self.model_registry.register_model(
                model_name=model_name,
                architecture=model_arch,
                dataset_name=config.dataset.value,
                experiment_id=config.experiment_id,
                parameters={
                    "preprocessing": config.preprocessing_config.name if config.preprocessing_config else "default",
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate
                },
                metrics=test_metrics,
                checkpoint_path=results_manager.checkpoints_dir / "best_model.pth"
            )
            
            # Generate experiment summary
            summary = results_manager.generate_experiment_summary()
            
            logger.info(f"Experiment completed: {config.experiment_name}")
            logger.info(f"Results saved to: {config.results_dir}")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}")
            results_manager.log_event("experiment_error", {"error": str(e)})
            raise
    
    def _run_cross_dataset_experiment(self, config: ExperimentConfig, 
                                    results_manager: ResultsManager) -> Dict[str, Any]:
        """Run an experiment testing models trained on one dataset against the other."""
        try:
            # Create a summary dictionary to store results
            cross_dataset_summary = {
                "experiment_id": config.experiment_id,
                "experiment_name": config.experiment_name,
                "model_architecture": config.model_architecture.value,
                "datasets": ["dataset1", "dataset2"],
                "results": {}
            }
            
            # Run experiments for each dataset
            for dataset in [ExperimentConfig.Dataset.DATASET1, ExperimentConfig.Dataset.DATASET2]:
                # Create a new config for this dataset
                dataset_config = ExperimentConfig(
                    experiment_id=f"{config.experiment_id}_{dataset.value}",
                    experiment_name=f"{config.experiment_name} - {dataset.value}",
                    dataset=dataset,
                    model_architecture=config.model_architecture,
                    preprocessing_config=config.preprocessing_config,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    cross_dataset_testing=False,
                    results_dir=str(config.results_dir / dataset.value)
                )
                
                # Run experiment for this dataset
                logger.info(f"Running experiment on {dataset.value}...")
                summary = self._run_single_model_experiment(dataset_config, ResultsManager(dataset_config))
                
                # Store results
                cross_dataset_summary["results"][dataset.value] = {
                    "training_summary": summary
                }
                
                # Now test on the other dataset
                other_dataset = ExperimentConfig.Dataset.DATASET1 if dataset == ExperimentConfig.Dataset.DATASET2 else ExperimentConfig.Dataset.DATASET2
                
                logger.info(f"Testing model trained on {dataset.value} against {other_dataset.value}...")
                
                # Get the best model checkpoint
                model_checkpoint = dataset_config.results_dir / "checkpoints" / "best_model.pth"
                
                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = get_model(config.model_architecture.value)
                model.load_state_dict(torch.load(model_checkpoint, map_location=device))
                model.to(device)
                model.eval()
                
                # Setup test loader for other dataset
                _, _, test_loader, class_names = self._setup_data_loaders(
                    other_dataset.value, 
                    config.preprocessing_config, 
                    batch_size=config.batch_size
                )
                
                # Test the model
                all_labels = []
                all_preds = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        if config.model_architecture.value == 'siamese':
                            output1, output2 = model(inputs[0], inputs[1])
                            preds = (F.pairwise_distance(output1, output2) < 0.5).float()
                        elif config.model_architecture.value == 'arcface':
                            embeddings = model(inputs)
                            outputs = F.linear(
                                F.normalize(embeddings), 
                                F.normalize(model.arcface.weight)
                            )
                            _, preds = torch.max(outputs, 1)
                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                cross_test_metrics = {
                    "accuracy": accuracy_score(all_labels, all_preds) * 100,
                    "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
                    "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
                    "f1_score": f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                }
                
                # Store cross-dataset results
                cross_dataset_summary["results"][dataset.value]["cross_dataset_testing"] = {
                    "tested_on": other_dataset.value,
                    "metrics": cross_test_metrics
                }
                
                # Log results
                logger.info(f"Cross-dataset testing results for model trained on {dataset.value}:")
                logger.info(f"Accuracy on {other_dataset.value}: {cross_test_metrics['accuracy']:.2f}%")
            
            # Save cross-dataset summary
            cross_summary_path = config.results_dir / "cross_dataset_summary.json"
            with open(cross_summary_path, 'w') as f:
                json.dump(cross_dataset_summary, f, indent=2)
            
            # Log completion
            logger.info(f"Cross-dataset experiment completed: {config.experiment_name}")
            logger.info(f"Results saved to: {config.results_dir}")
            
            return cross_dataset_summary
        
        except Exception as e:
            logger.error(f"Error running cross-dataset experiment: {str(e)}")
            results_manager.log_event("experiment_error", {"error": str(e)})
            raise
    
    def _run_architecture_comparison_experiment(self, config: ExperimentConfig, 
                                             results_manager: ResultsManager) -> Dict[str, Any]:
        """Run an experiment comparing different model architectures."""
        try:
            # Create a summary dictionary to store results
            comparison_summary = {
                "experiment_id": config.experiment_id,
                "experiment_name": config.experiment_name,
                "dataset": config.dataset.value,
                "architectures": [],
                "results": {}
            }
            
            # Determine which architectures to compare
            if isinstance(config.model_architecture.value, list):
                architectures = config.model_architecture.value
            else:
                # If not specified, compare all architectures
                architectures = [arch.value for arch in ExperimentConfig.ModelArchitecture]
            
            comparison_summary["architectures"] = architectures
            
            # Run experiments for each architecture
            for arch in architectures:
                # Create a new config for this architecture
                arch_config = ExperimentConfig(
                    experiment_id=f"{config.experiment_id}_{arch}",
                    experiment_name=f"{config.experiment_name} - {arch}",
                    dataset=config.dataset,
                    model_architecture=arch,
                    preprocessing_config=config.preprocessing_config,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    cross_dataset_testing=False,
                    results_dir=str(config.results_dir / arch)
                )
                
                # Run experiment for this architecture
                logger.info(f"Running experiment with {arch} architecture...")
                summary = self._run_single_model_experiment(arch_config, ResultsManager(arch_config))
                
                # Store results
                comparison_summary["results"][arch] = summary
            
            # Create comparison visualization
            self._create_architecture_comparison_plot(comparison_summary, config.results_dir)
            
            # Save comparison summary
            comparison_path = config.results_dir / "architecture_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison_summary, f, indent=2)
            
            # Log completion
            logger.info(f"Architecture comparison experiment completed: {config.experiment_name}")
            logger.info(f"Results saved to: {config.results_dir}")
            
            return comparison_summary
        
        except Exception as e:
            logger.error(f"Error running architecture comparison experiment: {str(e)}")
            results_manager.log_event("experiment_error", {"error": str(e)})
            raise
    
    def _create_architecture_comparison_plot(self, comparison_summary: Dict[str, Any], 
                                          output_dir: Path):
        """Create visualization comparing performance of different architectures."""
        architectures = comparison_summary["architectures"]
        
        # Extract metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for arch in architectures:
            if arch in comparison_summary["results"]:
                result = comparison_summary["results"][arch]
                
                # Get test metrics
                if "test_metrics" in result and result["test_metrics"]:
                    metrics = result["test_metrics"][0]  # Get first test metrics entry
                    accuracies.append(metrics.get("accuracy", 0))
                    precisions.append(metrics.get("precision", 0))
                    recalls.append(metrics.get("recall", 0))
                    f1_scores.append(metrics.get("f1_score", 0))
                else:
                    # If no test metrics, add zeros
                    accuracies.append(0)
                    precisions.append(0)
                    recalls.append(0)
                    f1_scores.append(0)
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "Architecture": architectures,
            "Accuracy": accuracies,
            "Precision": precisions,
            "Recall": recalls,
            "F1 Score": f1_scores
        })
        
        # Save to CSV
        df.to_csv(output_dir / "architecture_comparison.csv", index=False)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        bar_width = 0.2
        positions = np.arange(len(architectures))
        
        plt.bar(positions - bar_width*1.5, accuracies, bar_width, label='Accuracy', alpha=0.8)
        plt.bar(positions - bar_width/2, precisions, bar_width, label='Precision', alpha=0.8)
        plt.bar(positions + bar_width/2, recalls, bar_width, label='Recall', alpha=0.8)
        plt.bar(positions + bar_width*1.5, f1_scores, bar_width, label='F1 Score', alpha=0.8)
        
        plt.xlabel('Architecture')
        plt.ylabel('Score')
        plt.title('Performance Comparison Across Architectures')
        plt.xticks(positions, architectures)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "architecture_comparison.png")
        plt.close()
    
    def _setup_data_loaders(self, dataset_name: str, preprocessing_config: PreprocessingConfig, 
                         batch_size: int = 32):
        """Set up data loaders for training, validation and testing."""
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # If preprocessing config is provided, use the corresponding processed data
        if preprocessing_config:
            data_dir = Path(f"processed_data/{preprocessing_config.name}/{dataset_name}")
        else:
            # Otherwise, use default processed data
            data_dir = Path(f"processed_data/default/{dataset_name}")
        
        # Check if directory exists
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Load datasets
        train_dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
        val_dataset = datasets.ImageFolder(data_dir / "val", transform=transform)
        test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader, train_dataset.classes 


class DatasetComparisonExperiment:
    """Experiment for comparing performance between datasets."""
    
    def __init__(self, experiment_manager: ExperimentManager, 
                model_architecture: str = "cnn", 
                preprocessing_config: Optional[PreprocessingConfig] = None):
        """Initialize dataset comparison experiment."""
        self.experiment_manager = experiment_manager
        self.model_architecture = model_architecture
        self.preprocessing_config = preprocessing_config
        
    def run(self, epochs: int = 30, batch_size: int = 32, 
           learning_rate: float = 0.001) -> Dict[str, Any]:
        """Run dataset comparison experiment."""
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_name=f"Dataset Comparison - {self.model_architecture}",
            dataset=ExperimentConfig.Dataset.BOTH,
            model_architecture=self.model_architecture,
            preprocessing_config=self.preprocessing_config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cross_dataset_testing=True
        )
        
        # Run experiment
        return self.experiment_manager.run_experiment(config)
    
    def analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """Analyze and visualize dataset comparison results."""
        # Load experiment results
        try:
            with open(results_dir / "cross_dataset_summary.json", 'r') as f:
                summary = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Cross-dataset summary not found in {results_dir}")
        
        # Extract metrics for comparison
        dataset1_metrics = summary["results"]["dataset1"]["training_summary"]["test_metrics"][0]
        dataset2_metrics = summary["results"]["dataset2"]["training_summary"]["test_metrics"][0]
        dataset1_on_2_metrics = summary["results"]["dataset1"]["cross_dataset_testing"]["metrics"]
        dataset2_on_1_metrics = summary["results"]["dataset2"]["cross_dataset_testing"]["metrics"]
        
        # Create comparison table
        comparison = {
            "dataset1_same_domain": dataset1_metrics,
            "dataset2_same_domain": dataset2_metrics,
            "dataset1_cross_domain": dataset1_on_2_metrics,
            "dataset2_cross_domain": dataset2_on_1_metrics
        }
        
        # Create visualization
        self._create_dataset_comparison_plot(comparison, results_dir)
        
        # Analyze dataset statistics
        dataset_stats = self._analyze_dataset_statistics()
        
        # Combine results
        analysis = {
            "comparison": comparison,
            "dataset_statistics": dataset_stats,
            "insights": self._generate_insights(comparison, dataset_stats)
        }
        
        # Save analysis
        with open(results_dir / "dataset_comparison_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _create_dataset_comparison_plot(self, comparison: Dict[str, Dict[str, float]], 
                                     results_dir: Path):
        """Create visualization comparing performance across datasets."""
        # Extract accuracy metrics
        accuracies = [
            comparison["dataset1_same_domain"]["accuracy"],
            comparison["dataset1_cross_domain"]["accuracy"],
            comparison["dataset2_same_domain"]["accuracy"],
            comparison["dataset2_cross_domain"]["accuracy"]
        ]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        bar_labels = [
            "Dataset 1\nSame Domain",
            "Dataset 1\nCross Domain",
            "Dataset 2\nSame Domain",
            "Dataset 2\nCross Domain"
        ]
        
        bars = plt.bar(bar_labels, accuracies, color=['blue', 'lightblue', 'red', 'lightcoral'])
        
        plt.ylabel('Accuracy (%)')
        plt.title('Cross-Dataset Performance Comparison')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "dataset_comparison.png")
        plt.close()
        
        # Create a more detailed metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.2
        
        plt.bar(x - width*1.5, [comparison["dataset1_same_domain"][m] for m in metrics], 
              width, label='Dataset 1 Same Domain', color='blue')
        plt.bar(x - width/2, [comparison["dataset1_cross_domain"][m] for m in metrics], 
              width, label='Dataset 1 Cross Domain', color='lightblue')
        plt.bar(x + width/2, [comparison["dataset2_same_domain"][m] for m in metrics], 
              width, label='Dataset 2 Same Domain', color='red')
        plt.bar(x + width*1.5, [comparison["dataset2_cross_domain"][m] for m in metrics], 
              width, label='Dataset 2 Cross Domain', color='lightcoral')
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Detailed Cross-Dataset Performance Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "dataset_comparison_detailed.png")
        plt.close()
    
    def _analyze_dataset_statistics(self) -> Dict[str, Any]:
        """Analyze and compare dataset statistics."""
        # This could include:
        # - Number of images per dataset
        # - Number of subjects
        # - Images per subject
        # - Image quality metrics
        # - Face detection success rates
        # - etc.
        
        # For the sake of this implementation, we'll return some dummy data
        # In a real implementation, you would analyze the actual datasets
        return {
            "dataset1": {
                "num_subjects": 36,
                "images_per_subject": 49,
                "total_images": 36 * 49,
                "image_quality": "varied"
            },
            "dataset2": {
                "num_subjects": 18,
                "images_per_subject": 100,
                "total_images": 18 * 100,
                "image_quality": "consistent"
            }
        }
    
    def _generate_insights(self, comparison: Dict[str, Dict[str, float]], 
                         dataset_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights from dataset comparison."""
        insights = []
        
        # Compare same-domain performance
        dataset1_accuracy = comparison["dataset1_same_domain"]["accuracy"]
        dataset2_accuracy = comparison["dataset2_same_domain"]["accuracy"]
        
        if dataset1_accuracy > dataset2_accuracy:
            insights.append(f"Dataset 1 (high diversity) shows better in-domain performance than Dataset 2 (high quantity) by {dataset1_accuracy - dataset2_accuracy:.1f}%.")
        else:
            insights.append(f"Dataset 2 (high quantity) shows better in-domain performance than Dataset 1 (high diversity) by {dataset2_accuracy - dataset1_accuracy:.1f}%.")
        
        # Compare cross-domain generalization
        dataset1_cross = comparison["dataset1_cross_domain"]["accuracy"]
        dataset2_cross = comparison["dataset2_cross_domain"]["accuracy"]
        
        if dataset1_cross > dataset2_cross:
            insights.append(f"Models trained on Dataset 1 (high diversity) generalize better to new datasets by {dataset1_cross - dataset2_cross:.1f}%.")
        else:
            insights.append(f"Models trained on Dataset 2 (high quantity) generalize better to new datasets by {dataset2_cross - dataset1_cross:.1f}%.")
        
        # Compare performance drop
        drop1 = dataset1_accuracy - dataset1_cross
        drop2 = dataset2_accuracy - dataset2_cross
        
        if drop1 < drop2:
            insights.append(f"Models trained on Dataset 1 (high diversity) show more robust cross-domain performance with {drop2 - drop1:.1f}% less performance drop.")
        else:
            insights.append(f"Models trained on Dataset 2 (high quantity) show more robust cross-domain performance with {drop1 - drop2:.1f}% less performance drop.")
        
        return insights


class CrossArchitectureExperiment:
    """Experiment for comparing different model architectures."""
    
    def __init__(self, experiment_manager: ExperimentManager, 
                architectures: Optional[List[str]] = None,
                dataset: str = "both",
                preprocessing_config: Optional[PreprocessingConfig] = None):
        """Initialize architecture comparison experiment."""
        self.experiment_manager = experiment_manager
        
        # If architectures not specified, compare all available architectures
        if architectures is None:
            self.architectures = [arch.value for arch in ExperimentConfig.ModelArchitecture]
        else:
            self.architectures = architectures
            
        self.dataset = dataset
        self.preprocessing_config = preprocessing_config
        
    def run(self, epochs: int = 30, batch_size: int = 32, 
           learning_rate: float = 0.001) -> Dict[str, Any]:
        """Run architecture comparison experiment."""
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_name=f"Architecture Comparison - {self.dataset}",
            dataset=self.dataset,
            model_architecture=self.architectures,  # Pass all architectures to test
            preprocessing_config=self.preprocessing_config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cross_dataset_testing=False
        )
        
        # Run experiment
        return self.experiment_manager.run_experiment(config)
    
    def analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """Analyze and visualize architecture comparison results."""
        # Load experiment results
        try:
            with open(results_dir / "architecture_comparison.json", 'r') as f:
                summary = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Architecture comparison summary not found in {results_dir}")
        
        # Extract metrics for each architecture
        architectures = summary["architectures"]
        metrics = {}
        
        for arch in architectures:
            if arch in summary["results"]:
                result = summary["results"][arch]
                if "test_metrics" in result and result["test_metrics"]:
                    metrics[arch] = result["test_metrics"][0]
        
        # Create comprehensive comparison visualization
        self._create_architecture_comparison_plot(metrics, results_dir)
        
        # Create comparative performance matrix
        perf_matrix = self._create_performance_matrix(metrics)
        
        # Generate architecture insights
        insights = self._generate_architecture_insights(metrics)
        
        # Combine results
        analysis = {
            "metrics": metrics,
            "performance_matrix": perf_matrix,
            "insights": insights
        }
        
        # Save analysis
        with open(results_dir / "architecture_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _create_architecture_comparison_plot(self, metrics: Dict[str, Dict[str, float]], 
                                         results_dir: Path):
        """Create detailed visualization comparing architectures."""
        # Extract metric types
        metric_types = ['accuracy', 'precision', 'recall', 'f1_score']
        architectures = list(metrics.keys())
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot each metric type
        for i, metric in enumerate(metric_types):
            values = [metrics[arch].get(metric, 0) for arch in architectures]
            
            axes[i].bar(architectures, values, color='skyblue')
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel('Score')
            axes[i].set_xticklabels(architectures, rotation=45)
            
            # Add values on top of bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(results_dir / "architecture_comparison_detailed.png")
        plt.close()
        
        # Create a spider/radar chart for comparative visualization
        categories = metric_types
        fig = plt.figure(figsize=(10, 10))
        
        # Create radar chart
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angle for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each architecture
        for arch in architectures:
            values = [metrics[arch].get(metric, 0) for metric in categories]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=arch)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.title('Architecture Comparison - Radar Chart')
        plt.savefig(results_dir / "architecture_comparison_radar.png")
        plt.close()
    
    def _create_performance_matrix(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Create a performance matrix comparing architectures on key metrics."""
        architectures = list(metrics.keys())
        matrix = {}
        
        # Create a matrix with relative performance rankings
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            # Sort architectures by this metric
            sorted_archs = sorted(architectures, 
                                key=lambda arch: metrics[arch].get(metric, 0), 
                                reverse=True)
            
            # Assign rankings
            for i, arch in enumerate(sorted_archs):
                if arch not in matrix:
                    matrix[arch] = {}
                
                matrix[arch][metric] = i + 1  # 1-based ranking
        
        # Calculate overall ranking
        for arch in architectures:
            if arch in matrix:
                matrix[arch]['overall'] = sum([matrix[arch].get(m, len(architectures)) 
                                            for m in ['accuracy', 'precision', 'recall', 'f1_score']])
        
        return matrix
    
    def _generate_architecture_insights(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate insights from architecture comparison."""
        insights = []
        
        # Find best overall architecture
        if metrics:
            best_arch = max(metrics.keys(), key=lambda arch: metrics[arch].get('accuracy', 0))
            best_acc = metrics[best_arch].get('accuracy', 0)
            insights.append(f"{best_arch} achieves the highest accuracy at {best_acc:.2f}%.")
        
        # Compare architectures pairwise
        architectures = list(metrics.keys())
        for i, arch1 in enumerate(architectures):
            for arch2 in architectures[i+1:]:
                acc1 = metrics[arch1].get('accuracy', 0)
                acc2 = metrics[arch2].get('accuracy', 0)
                
                if abs(acc1 - acc2) > 1.0:  # Only report meaningful differences
                    better = arch1 if acc1 > acc2 else arch2
                    worse = arch2 if acc1 > acc2 else arch1
                    diff = abs(acc1 - acc2)
                    insights.append(f"{better} outperforms {worse} by {diff:.2f}% in accuracy.")
        
        # Identify strengths for each architecture
        for arch in architectures:
            # Find the metric where this architecture performs best relative to others
            best_metric = None
            best_rank = float('inf')
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                # Calculate the rank of this architecture for this metric
                values = [metrics[a].get(metric, 0) for a in architectures]
                sorted_values = sorted(values, reverse=True)
                rank = sorted_values.index(metrics[arch].get(metric, 0)) + 1
                
                if rank < best_rank:
                    best_rank = rank
                    best_metric = metric
            
            if best_metric and best_rank <= 2:  # Only highlight top performers
                insights.append(f"{arch} shows particular strength in {best_metric} ({metrics[arch].get(best_metric, 0):.2f}).")
        
        return insights


class HyperparameterExperiment:
    """Experiment for optimizing hyperparameters."""
    
    def __init__(self, experiment_manager: ExperimentManager,
                model_architecture: str = "cnn",
                dataset: str = "both",
                preprocessing_config: Optional[PreprocessingConfig] = None,
                n_trials: int = 20,
                timeout: int = 7200):  # 2 hours default timeout
        """Initialize hyperparameter experiment."""
        self.experiment_manager = experiment_manager
        self.model_architecture = model_architecture
        self.dataset = dataset
        self.preprocessing_config = preprocessing_config
        self.n_trials = n_trials
        self.timeout = timeout
        
    def run(self) -> Dict[str, Any]:
        """Run hyperparameter optimization experiment."""
        # Create a timestamp for this optimization run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"hyperopt_{self.model_architecture}_{timestamp}"
        
        # Create results directory
        results_dir = OUT_DIR / experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"hyperopt_{self.model_architecture}",
            storage=f"sqlite:///{results_dir}/optuna.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, results_dir),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Save study results
        self._save_study_results(study, results_dir)
        
        # Get best parameters
        best_params = study.best_params
        
        # Run a final experiment with best parameters
        best_config = ExperimentConfig(
            experiment_id=f"{experiment_id}_best",
            experiment_name=f"Best Hyperparameters for {self.model_architecture}",
            dataset=self.dataset,
            model_architecture=self.model_architecture,
            preprocessing_config=self.preprocessing_config,
            epochs=best_params.get("epochs", 30),
            batch_size=best_params.get("batch_size", 32),
            learning_rate=best_params.get("learning_rate", 0.001),
            cross_dataset_testing=False,
            results_dir=str(results_dir / "best_params")
        )
        
        # Run experiment with best parameters
        final_results = self.experiment_manager.run_experiment(best_config)
        
        # Combine results
        summary = {
            "study_name": study.study_name,
            "num_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_params": best_params,
            "best_value": study.best_value,
            "final_results": final_results
        }
        
        # Save summary
        with open(results_dir / "hyperopt_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _objective(self, trial: optuna.Trial, results_dir: Path) -> float:
        """Objective function for hyperparameter optimization."""
        # Define hyperparameter space
        epochs = trial.suggest_int("epochs", 10, 50)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        
        # Additional hyperparameters can be added here
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id=f"trial_{trial.number}",
            experiment_name=f"Hyperopt Trial {trial.number}",
            dataset=self.dataset,
            model_architecture=self.model_architecture,
            preprocessing_config=self.preprocessing_config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cross_dataset_testing=False,
            results_dir=str(results_dir / f"trial_{trial.number}")
        )
        
        try:
            # Run experiment
            results = self.experiment_manager.run_experiment(config)
            
            # Extract validation accuracy from results
            if "best_validation_metrics" in results and "accuracy" in results["best_validation_metrics"]:
                return results["best_validation_metrics"]["accuracy"]
            else:
                # If validation metrics not found, use test accuracy
                if "test_metrics" in results and results["test_metrics"] and "accuracy" in results["test_metrics"][0]:
                    return results["test_metrics"][0]["accuracy"]
                else:
                    # If no metrics found, return a very low score
                    return 0.0
        
        except Exception as e:
            logger.error(f"Error in hyperopt trial {trial.number}: {str(e)}")
            return 0.0  # Return low score on error
    
    def _save_study_results(self, study: optuna.Study, results_dir: Path):
        """Save and visualize optimization study results."""
        # Save optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(results_dir / "optimization_history.png")
        plt.close()
        
        # Save parameter importances
        plt.figure(figsize=(10, 6))
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(results_dir / "param_importances.png")
        except:
            logger.warning("Could not plot parameter importances (requires more than one trial)")
        plt.close()
        
        # Save parallel coordinate plot
        plt.figure(figsize=(12, 8))
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.savefig(results_dir / "parallel_coordinate.png")
        except:
            logger.warning("Could not plot parallel coordinates (requires more than one trial)")
        plt.close()
        
        # Create optimization trials table
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    "number": trial.number,
                    "value": trial.value,
                    **trial.params
                }
                trials_data.append(trial_data)
        
        # Save trials data to CSV
        if trials_data:
            pd.DataFrame(trials_data).to_csv(results_dir / "trials.csv", index=False)


class ResultsCompiler:
    """Utility for creating presentation-ready reports and visualizations."""
    
    def __init__(self, base_results_dir: Optional[Path] = None):
        """Initialize results compiler."""
        self.base_results_dir = base_results_dir or OUT_DIR
    
    def compile_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Compile a comprehensive summary of an experiment."""
        # Find experiment directory
        experiment_dirs = list(self.base_results_dir.glob(f"*{experiment_id}*"))
        if not experiment_dirs:
            raise ValueError(f"No experiment found with ID: {experiment_id}")
        
        experiment_dir = experiment_dirs[0]
        
        # Load experiment configuration
        config_file = experiment_dir / "experiment_config.json"
        if not config_file.exists():
            config_file = next(experiment_dir.glob("**/experiment_config.json"), None)
            
        if not config_file:
            raise ValueError(f"No configuration file found for experiment: {experiment_id}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load experiment summary
        summary_file = experiment_dir / "experiment_summary.json"
        if not summary_file.exists():
            summary_file = next(experiment_dir.glob("**/experiment_summary.json"), None)
            
        summary = None
        if summary_file:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        # Find all visualization files
        vis_files = list(experiment_dir.glob("**/*.png"))
        
        # Create report structure
        report = {
            "experiment_id": experiment_id,
            "config": config,
            "summary": summary,
            "visualizations": [str(f.relative_to(self.base_results_dir)) for f in vis_files]
        }
        
        return report
    
    def generate_comparative_report(self, experiment_ids: List[str], 
                                   output_dir: Optional[Path] = None) -> Path:
        """Generate a comparative report of multiple experiments."""
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.base_results_dir / f"comparative_report_{timestamp}"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile summaries for all experiments
        experiment_summaries = []
        for exp_id in experiment_ids:
            try:
                summary = self.compile_experiment_summary(exp_id)
                experiment_summaries.append(summary)
            except ValueError as e:
                logger.warning(f"Error compiling summary for experiment {exp_id}: {str(e)}")
        
        # Create comparative visualizations
        self._create_comparative_visualizations(experiment_summaries, output_dir)
        
        # Create report content
        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "experiments": experiment_summaries,
            "comparative_visualizations": list(output_dir.glob("*.png"))
        }
        
        # Save report
        with open(output_dir / "comparative_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_dir
    
    def _create_comparative_visualizations(self, summaries: List[Dict[str, Any]], 
                                        output_dir: Path):
        """Create visualizations comparing multiple experiments."""
        # Extract experiment names and key metrics
        exp_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for summary in summaries:
            if "summary" in summary and summary["summary"] and "test_metrics" in summary["summary"]:
                test_metrics = summary["summary"]["test_metrics"]
                if test_metrics and len(test_metrics) > 0:
                    metrics = test_metrics[0]
                    
                    exp_names.append(summary["experiment_id"])
                    accuracies.append(metrics.get("accuracy", 0))
                    precisions.append(metrics.get("precision", 0))
                    recalls.append(metrics.get("recall", 0))
                    f1_scores.append(metrics.get("f1_score", 0))
        
        if not exp_names:
            logger.warning("No valid metrics found for comparative visualization")
            return
        
        # Create comparison bar chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(exp_names))
        width = 0.2
        
        plt.bar(x - width*1.5, accuracies, width, label='Accuracy')
        plt.bar(x - width/2, precisions, width, label='Precision')
        plt.bar(x + width/2, recalls, width, label='Recall')
        plt.bar(x + width*1.5, f1_scores, width, label='F1 Score')
        
        plt.xlabel('Experiment')
        plt.ylabel('Score')
        plt.title('Performance Comparison Across Experiments')
        plt.xticks(x, exp_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png")
        plt.close()
    
    def generate_powerpoint_report(self, experiment_ids: List[str], 
                                 output_path: Optional[Path] = None) -> Path:
        """Generate a PowerPoint presentation with experiment results."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
        except ImportError:
            raise ImportError("python-pptx is required for PowerPoint generation. Install with: pip install python-pptx")
        
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_results_dir / f"experiment_report_{timestamp}.pptx"
        
        # Create a presentation
        prs = Presentation()
        
        # Add title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        title.text = "Face Recognition Experiment Results"
        subtitle.text = f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Compile summaries for all experiments
        for exp_id in experiment_ids:
            try:
                summary = self.compile_experiment_summary(exp_id)
                
                # Add experiment slide
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                title = slide.shapes.title
                title.text = f"Experiment: {summary['experiment_id']}"
                
                # Add experiment details
                if "config" in summary:
                    config = summary["config"]
                    content = slide.placeholders[1]
                    tf = content.text_frame
                    tf.text = f"Model: {config.get('model_architecture', 'N/A')}\n"
                    tf.text += f"Dataset: {config.get('dataset', 'N/A')}\n"
                    
                    if "summary" in summary and summary["summary"]:
                        s = summary["summary"]
                        if "test_metrics" in s and s["test_metrics"] and len(s["test_metrics"]) > 0:
                            metrics = s["test_metrics"][0]
                            tf.text += f"\nResults:\n"
                            tf.text += f"Accuracy: {metrics.get('accuracy', 0):.2f}%\n"
                            tf.text += f"Precision: {metrics.get('precision', 0):.2f}\n"
                            tf.text += f"Recall: {metrics.get('recall', 0):.2f}\n"
                            tf.text += f"F1 Score: {metrics.get('f1_score', 0):.2f}\n"
                
                # Add visualization slides
                for vis in summary.get("visualizations", [])[:3]:  # Limit to 3 visualizations
                    vis_path = self.base_results_dir / vis
                    if vis_path.exists():
                        slide = prs.slides.add_slide(prs.slide_layouts[6])
                        title = slide.shapes.title
                        title.text = f"Visualization: {vis_path.stem}"
                        
                        # Add image
                        slide.shapes.add_picture(str(vis_path), Inches(1), Inches(1.5), 
                                             width=Inches(8), height=Inches(5))
            
            except ValueError as e:
                logger.warning(f"Error compiling summary for experiment {exp_id}: {str(e)}")
                continue
        
        # Save presentation
        prs.save(output_path)
        
        return output_path
    
    def export_to_excel(self, experiment_ids: List[str], 
                      output_path: Optional[Path] = None) -> Path:
        """Export experiment results to Excel spreadsheet."""
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_results_dir / f"experiment_results_{timestamp}.xlsx"
        
        # Compile results for all experiments
        results = []
        
        for exp_id in experiment_ids:
            try:
                summary = self.compile_experiment_summary(exp_id)
                
                if "config" in summary and "summary" in summary and summary["summary"]:
                    config = summary["config"]
                    s = summary["summary"]
                    
                    result = {
                        "experiment_id": exp_id,
                        "model_architecture": config.get("model_architecture", ""),
                        "dataset": config.get("dataset", ""),
                        "epochs": config.get("epochs", ""),
                        "batch_size": config.get("batch_size", ""),
                        "learning_rate": config.get("learning_rate", "")
                    }
                    
                    # Add test metrics
                    if "test_metrics" in s and s["test_metrics"] and len(s["test_metrics"]) > 0:
                        metrics = s["test_metrics"][0]
                        result.update({
                            "accuracy": metrics.get("accuracy", ""),
                            "precision": metrics.get("precision", ""),
                            "recall": metrics.get("recall", ""),
                            "f1_score": metrics.get("f1_score", "")
                        })
                    
                    results.append(result)
            
            except ValueError as e:
                logger.warning(f"Error compiling summary for experiment {exp_id}: {str(e)}")
                continue
        
        # Create Excel file
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_path, index=False)
        
        return output_path


def create_command_line_interface():
    """Create a command-line interface for the experiment manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Experiment Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dataset Comparison experiment
    parser_dataset = subparsers.add_parser("dataset-comparison", 
                                          help="Run dataset comparison experiment")
    parser_dataset.add_argument("--architecture", type=str, default="cnn",
                              help="Model architecture to use")
    parser_dataset.add_argument("--epochs", type=int, default=30,
                              help="Number of training epochs")
    parser_dataset.add_argument("--batch-size", type=int, default=32,
                              help="Batch size for training")
    parser_dataset.add_argument("--learning-rate", type=float, default=0.001,
                              help="Learning rate for training")
    
    # Architecture Comparison experiment
    parser_arch = subparsers.add_parser("architecture-comparison", 
                                      help="Run architecture comparison experiment")
    parser_arch.add_argument("--dataset", type=str, default="both",
                           help="Dataset to use (dataset1, dataset2, or both)")
    parser_arch.add_argument("--architectures", type=str, nargs="+",
                           default=["baseline", "cnn", "attention", "arcface", "hybrid"],
                           help="Model architectures to compare")
    parser_arch.add_argument("--epochs", type=int, default=30,
                           help="Number of training epochs")
    parser_arch.add_argument("--batch-size", type=int, default=32,
                           help="Batch size for training")
    parser_arch.add_argument("--learning-rate", type=float, default=0.001,
                           help="Learning rate for training")
    
    # Hyperparameter Optimization experiment
    parser_hyperopt = subparsers.add_parser("hyperparameter-optimization",
                                          help="Run hyperparameter optimization experiment")
    parser_hyperopt.add_argument("--architecture", type=str, default="cnn",
                               help="Model architecture to optimize")
    parser_hyperopt.add_argument("--dataset", type=str, default="both",
                               help="Dataset to use (dataset1, dataset2, or both)")
    parser_hyperopt.add_argument("--n-trials", type=int, default=20,
                               help="Number of optimization trials")
    parser_hyperopt.add_argument("--timeout", type=int, default=7200,
                               help="Timeout in seconds")
    
    # Generate report
    parser_report = subparsers.add_parser("generate-report",
                                        help="Generate report from experiment results")
    parser_report.add_argument("--experiment-ids", type=str, nargs="+", required=True,
                             help="Experiment IDs to include in report")
    parser_report.add_argument("--format", type=str, choices=["json", "pptx", "excel"],
                             default="json", help="Report format")
    parser_report.add_argument("--output", type=str, help="Output file path")
    
    # Resume experiment
    parser_resume = subparsers.add_parser("resume",
                                        help="Resume an interrupted experiment")
    parser_resume.add_argument("experiment_id", type=str,
                             help="ID of experiment to resume")
    
    return parser


def main():
    parser = create_command_line_interface()
    args = parser.parse_args()

    if args.command == "dataset-comparison":
        experiment_manager = ExperimentManager()
        results = DatasetComparisonExperiment(experiment_manager).run(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        print(f"Dataset Comparison Results: {results}")
    elif args.command == "architecture-comparison":
        experiment_manager = ExperimentManager()
        results = CrossArchitectureExperiment(experiment_manager, args.architectures, args.dataset).run(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        print(f"Architecture Comparison Results: {results}")
    elif args.command == "hyperparameter-optimization":
        experiment_manager = ExperimentManager()
        results = HyperparameterExperiment(experiment_manager).run()
        print(f"Hyperparameter Optimization Results: {results}")
    elif args.command == "generate-report":
        results_compiler = ResultsCompiler()
        if args.format == "json":
            results_compiler.generate_comparative_report(args.experiment_ids, args.output)
        elif args.format == "pptx":
            results_compiler.generate_powerpoint_report(args.experiment_ids, args.output)
        elif args.format == "excel":
            results_compiler.export_to_excel(args.experiment_ids, args.output)
    elif args.command == "resume":
        experiment_manager = ExperimentManager()
        results = experiment_manager.run_experiment(ExperimentConfig(experiment_id=args.experiment_id))
        print(f"Resumed Experiment Results: {results}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 