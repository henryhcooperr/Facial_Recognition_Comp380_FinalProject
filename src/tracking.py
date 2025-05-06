#!/usr/bin/env python3

import os
import json
import logging
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt

from .base_config import logger


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking systems.
    Made this after spending WAY too many hours trying to decide between
    different tracking systems... this way I can swap them easily.
    """
    
    @abstractmethod
    def initialize(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        """
        Sets up the tracking system.
        
        Args:
            experiment_name: What to call this experiment
            tracking_uri: Where to find the tracking server (if needed)
        """
        pass
    
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, run_id: Optional[str] = None,
                tags: Optional[Dict[str, str]] = None) -> str:
        """Start tracking a new experiment run."""
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """Finish the current tracking run."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step or iteration number
        """
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Track a file as an artifact."""
        pass
    
    @abstractmethod
    def log_figure(self, figure: plt.Figure, artifact_name: str) -> None:
        """Save a matplotlib figure."""
        pass
    
    @abstractmethod
    def log_model(self, model: torch.nn.Module, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a PyTorch model."""
        pass
    
    @abstractmethod
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], name: str = "confusion_matrix") -> None:
        """Track a confusion matrix."""
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Add tags to the current run."""
        pass
    
    @abstractmethod
    def get_dashboard_url(self) -> Optional[str]:
        """Get a URL to view this run in the web UI."""
        pass
    
    @staticmethod
    def create(tracker_type: str, **kwargs) -> 'ExperimentTracker':
        """
        Factory method to create a tracker.
        
        This is a neat trick I learned - factories are awesome!
        
        Args:
            tracker_type: Type of tracker ('mlflow', 'wandb', or 'none')
            **kwargs: Extra args for the tracker
            
        Returns:
            A tracker instance
        """
        if tracker_type.lower() == 'mlflow':
            return MLflowTracker(**kwargs)
        elif tracker_type.lower() in ('wandb', 'weights_and_biases'):
            return WeightsAndBiasesTracker(**kwargs)
        elif tracker_type.lower() == 'none':
            return NoopTracker(**kwargs)
        else:
            logger.warning(f"Unknown tracker type: {tracker_type}. Using NoopTracker.")
            return NoopTracker(**kwargs)
    
    @abstractmethod
    def compare_runs(self, run_ids: List[str], metric_names: List[str], 
                   output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
        """Compare multiple runs on a chart."""
        pass
    
    @abstractmethod
    def search_runs(self, filter_string: str, 
                  max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for runs that match some criteria."""
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow implementation of experiment tracking."""
    
    def __init__(self, **kwargs):
        """Set up MLflow tracker."""
        self.experiment_id = None
        self.run_id = None
        self.active = False
        
        # Lazy import MLflow to avoid dependency if not used
        # This is a fancy trick I learned to make dependencies optional
        try:
            import mlflow
            self.mlflow = mlflow
            self.available = True
        except ImportError:
            logger.warning("MLflow not installed. Please run: pip install mlflow")
            self.available = False
    
    def initialize(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        """Get MLflow ready with the given experiment name and tracking URI."""
        if not self.available:
            logger.warning("MLflow not available. Skipping initialization.")
            return
            
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
            
        # Get or create the experiment
        try:
            experiment = self.mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = self.mlflow.create_experiment(experiment_name)
                
            logger.info(f"MLflow initialized with experiment: {experiment_name} (ID: {self.experiment_id})")
        except Exception as e:
            logger.error(f"Error initializing MLflow: {str(e)}")
            self.available = False
    
    def start_run(self, run_name: Optional[str] = None, run_id: Optional[str] = None,
                tags: Optional[Dict[str, str]] = None) -> str:
        """Begin a new MLflow run."""
        if not self.available:
            logger.warning("MLflow not available. Skipping start_run.")
            return ""
            
        try:
            # TODO: Fix this so we can resume runs better
            # Spent 2 hours debugging this on 5/20 and still not working right
            active_run = self.mlflow.start_run(
                run_id=run_id,
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            self.run_id = active_run.info.run_id
            self.active = True
            logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
            return self.run_id
        except Exception as e:
            logger.error(f"Error starting MLflow run: {str(e)}")
            return ""
    
    def end_run(self) -> None:
        """Finish the current MLflow run."""
        if not self.available or not self.active:
            return
            
        try:
            self.mlflow.end_run()
            self.active = False
            logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.available or not self.active:
            return
            
        try:
            # Convert complex types to strings to ensure they can be logged
            serialized_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list, tuple, set)):
                    serialized_params[key] = json.dumps(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    serialized_params[key] = value
                else:
                    # Just convert everything else to string
                    # Not elegant but it works -HC 5/22/23
                    serialized_params[key] = str(value)
                    
            self.mlflow.log_params(serialized_params)
        except Exception as e:
            logger.error(f"Error logging parameters to MLflow: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics to MLflow."""
        if not self.available or not self.active:
            return
            
        try:
            self.mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {str(e)}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Save a file to MLflow."""
        if not self.available or not self.active:
            return
            
        try:
            self.mlflow.log_artifact(str(local_path), artifact_path)
        except Exception as e:
            logger.error(f"Error logging artifact to MLflow: {str(e)}")
    
    def log_figure(self, figure: plt.Figure, artifact_name: str) -> None:
        """Save a matplotlib figure to MLflow."""
        if not self.available or not self.active:
            return
            
        try:
            # Save figure to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                figure.savefig(tmp.name, bbox_inches='tight')
                tmp_name = tmp.name
                
            # Log the temporary file as an artifact
            self.mlflow.log_artifact(tmp_name, artifact_name)
            
            # Remove the temporary file
            os.remove(tmp_name)
        except Exception as e:
            logger.error(f"Error logging figure to MLflow: {str(e)}")
    
    def log_model(self, model: torch.nn.Module, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a PyTorch model to MLflow."""
        if not self.available or not self.active:
            return
            
        try:
            # Log the model with MLflow's built-in PyTorch flavor
            self.mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                registered_model_name=model_name if metadata and metadata.get('register', False) else None,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {str(e)}")
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], name: str = "confusion_matrix") -> None:
        """Create and save a confusion matrix visualization."""
        if not self.available or not self.active:
            return
            
        try:
            # Create and save figure
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                 yticks=np.arange(cm.shape[0]),
                 xticklabels=class_names, yticklabels=class_names,
                 ylabel='True label',
                 xlabel='Predicted label')
            
            # Rotate tick labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations
            fmt = 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            
            # Log the figure
            self.log_figure(fig, f"{name}.png")
            plt.close(fig)
            
            # Also log the matrix as a JSON file
            import tempfile
            import json
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                with open(tmp.name, 'w') as f:
                    json.dump({
                        "matrix": cm.tolist(),
                        "class_names": class_names
                    }, f)
                tmp_name = tmp.name
                
            self.mlflow.log_artifact(tmp_name, f"{name}.json")
            os.remove(tmp_name)
        except Exception as e:
            logger.error(f"Error logging confusion matrix to MLflow: {str(e)}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Add tags to the current MLflow run."""
        if not self.available or not self.active:
            return
            
        try:
            self.mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Error setting tags in MLflow: {str(e)}")
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the URL to the MLflow dashboard for the current run."""
        if not self.available or not self.active:
            return None
            
        try:
            tracking_uri = self.mlflow.get_tracking_uri()
            if tracking_uri.startswith("http"):
                return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
            return None
        except Exception as e:
            logger.error(f"Error getting MLflow dashboard URL: {str(e)}")
            return None
    
    # FIXME: This doesn't handle missing metrics well
    # (6/4/23) Had to hack this together for my demo
    def compare_runs(self, run_ids: List[str], metric_names: List[str], 
                   output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
        """Create a comparison visualization for multiple MLflow runs."""
        if not self.available:
            logger.warning("MLflow not available. Skipping compare_runs.")
            return None
            
        try:
            # Fetch run data
            client = self.mlflow.tracking.MlflowClient()
            runs = [client.get_run(run_id) for run_id in run_ids]
            
            # Extract run names or IDs and metrics
            run_names = [run.data.tags.get("mlflow.runName", run.info.run_id) for run in runs]
            metric_values = {}
            
            for metric in metric_names:
                metric_values[metric] = []
                for run in runs:
                    # Check if the metric exists for this run
                    metric_value = next((m.value for m in client.get_metric_history(run.info.run_id, metric)), None)
                    metric_values[metric].append(metric_value)
            
            # Create visualization
            fig, axes = plt.subplots(nrows=len(metric_names), figsize=(12, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
                
            for i, metric in enumerate(metric_names):
                values = metric_values[metric]
                if all(v is not None for v in values):
                    axes[i].bar(run_names, values)
                    axes[i].set_title(f"Comparison of {metric}")
                    axes[i].set_ylabel(metric)
                    plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")
                else:
                    axes[i].text(0.5, 0.5, f"No data available for metric: {metric}", 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                fig.savefig(output_path, bbox_inches='tight')
            
            return fig
        except Exception as e:
            logger.error(f"Error comparing MLflow runs: {str(e)}")
            return None
    
    def search_runs(self, filter_string: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Find MLflow runs matching the filter criteria."""
        if not self.available:
            logger.warning("MLflow not available. Skipping search_runs.")
            return []
            
        try:
            # Use MLflow's search runs capability
            runs = self.mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            # Convert DataFrame to list of dicts
            return runs.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching MLflow runs: {str(e)}")
            return []


class WeightsAndBiasesTracker(ExperimentTracker):
    """Weights & Biases implementation of experiment tracking."""
    
    def __init__(self, **kwargs):
        """Initialize Weights & Biases tracker."""
        self.run = None
        self.active = False
        self.project_name = kwargs.get('project', 'face-recognition')
        self.entity = kwargs.get('entity', None)
        
        # Lazy import Weights & Biases to avoid dependency if not used
        try:
            import wandb
            self.wandb = wandb
            self.available = True
        except ImportError:
            logger.warning("Weights & Biases not installed. Please run: pip install wandb")
            self.available = False
    
    def initialize(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        """Initialize Weights & Biases with the given project name."""
        if not self.available:
            logger.warning("Weights & Biases not available. Skipping initialization.")
            return
            
        # W&B doesn't need explicit initialization, but we can set the project name
        self.project_name = experiment_name
        logger.info(f"Weights & Biases initialized with project: {self.project_name}")
    
    def start_run(self, run_name: Optional[str] = None, run_id: Optional[str] = None,
                tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new Weights & Biases run."""
        if not self.available:
            logger.warning("Weights & Biases not available. Skipping start_run.")
            return ""
            
        try:
            # Set default tags if none provided 
            # Had issues with None tags - HC 6/1/23
            wandb_tags = tags or []
            
            # Initialize W&B run
            self.run = self.wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                id=run_id,
                resume="allow" if run_id else None,
                tags=wandb_tags,
                reinit=True
            )
            self.active = True
            logger.info(f"Started W&B run: {run_name} (ID: {self.run.id})")
            return self.run.id
        except Exception as e:
            logger.error(f"Error starting W&B run: {str(e)}")
            return ""
    
    def end_run(self) -> None:
        """End the current Weights & Biases run."""
        if not self.available or not self.active:
            return
            
        try:
            self.wandb.finish()
            self.active = False
            logger.info(f"Ended W&B run: {self.run.id if self.run else 'Unknown'}")
            self.run = None
        except Exception as e:
            logger.error(f"Error ending W&B run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            # W&B handles complex types well, so we can log directly
            self.wandb.config.update(params)
        except Exception as e:
            logger.error(f"Error logging parameters to W&B: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics to W&B: {str(e)}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log an artifact to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            # Create an artifact
            artifact_name = artifact_path or os.path.basename(local_path)
            artifact = self.wandb.Artifact(name=artifact_name, type="dataset")
            artifact.add_file(str(local_path))
            self.run.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Error logging artifact to W&B: {str(e)}")
    
    def log_figure(self, figure: plt.Figure, artifact_name: str) -> None:
        """Save a matplotlib figure to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            # W&B has direct support for matplotlib figures
            self.wandb.log({artifact_name: self.wandb.Image(figure)})
        except Exception as e:
            logger.error(f"Error logging figure to W&B: {str(e)}")
    
    def log_model(self, model: torch.nn.Module, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a PyTorch model to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            # Create an artifact for the model
            artifact = self.wandb.Artifact(name=model_name, type="model", metadata=metadata)
            
            # Save the model to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                torch.save(model.state_dict(), tmp.name)
                tmp_name = tmp.name
            
            # Add the file to the artifact
            artifact.add_file(tmp_name, name="model.pt")
            
            # Log the artifact
            self.run.log_artifact(artifact)
            
            # Remove the temporary file
            os.remove(tmp_name)
        except Exception as e:
            logger.error(f"Error logging model to W&B: {str(e)}")
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], name: str = "confusion_matrix") -> None:
        """Log a confusion matrix to Weights & Biases."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            # W&B has a built-in confusion matrix feature
            self.wandb.log({name: self.wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.arange(len(class_names)),  # Just need indices for this to work
                preds=np.arange(len(class_names)),
                class_names=class_names,
                matrix=cm
            )})
        except Exception as e:
            logger.error(f"Error logging confusion matrix to W&B: {str(e)}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current Weights & Biases run."""
        if not self.available or not self.active or not self.run:
            return
            
        try:
            for key, value in tags.items():
                self.run.tags.append(f"{key}:{value}")
        except Exception as e:
            logger.error(f"Error setting tags in W&B: {str(e)}")
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the URL to the Weights & Biases dashboard for the current run."""
        if not self.available or not self.active or not self.run:
            return None
            
        try:
            return self.run.get_url()
        except Exception as e:
            logger.error(f"Error getting W&B dashboard URL: {str(e)}")
            return None
    
    def compare_runs(self, run_ids: List[str], metric_names: List[str], 
                   output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
        """Create a comparison visualization for multiple Weights & Biases runs."""
        if not self.available:
            logger.warning("Weights & Biases not available. Skipping compare_runs.")
            return None
            
        try:
            # Import the W&B API
            api = self.wandb.Api()
            
            # Get the runs
            runs = [api.run(f"{self.entity}/{self.project_name}/{run_id}") for run_id in run_ids]
            
            # Extract run names and metrics
            run_names = [run.name for run in runs]
            metric_values = {}
            
            for metric in metric_names:
                metric_values[metric] = []
                for run in runs:
                    # Get the summary metrics
                    if metric in run.summary:
                        metric_values[metric].append(run.summary[metric])
                    else:
                        metric_values[metric].append(None)
            
            # Create visualization
            fig, axes = plt.subplots(nrows=len(metric_names), figsize=(12, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
                
            for i, metric in enumerate(metric_names):
                values = metric_values[metric]
                if all(v is not None for v in values):
                    axes[i].bar(run_names, values)
                    axes[i].set_title(f"Comparison of {metric}")
                    axes[i].set_ylabel(metric)
                    plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")
                else:
                    axes[i].text(0.5, 0.5, f"No data available for metric: {metric}", 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                fig.savefig(output_path, bbox_inches='tight')
            
            return fig
        except Exception as e:
            logger.error(f"Error comparing W&B runs: {str(e)}")
            return None
    
    def search_runs(self, filter_string: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for Weights & Biases runs matching the filter criteria."""
        if not self.available:
            logger.warning("Weights & Biases not available. Skipping search_runs.")
            return []
            
        try:
            # Import the W&B API
            api = self.wandb.Api()
            
            # Parse the filter string into a W&B query format
            query = filter_string
            
            # Search for runs
            runs = api.runs(
                path=f"{self.entity}/{self.project_name}",
                filters={"$and": [{"config": query}]},
                per_page=max_results
            )
            
            # Convert the run objects to dictionaries
            results = []
            for run in runs:
                result = {
                    "id": run.id,
                    "name": run.name,
                    "summary": dict(run.summary),
                    "config": dict(run.config),
                    "url": run.url
                }
                results.append(result)
                
                if len(results) >= max_results:
                    break
                    
            return results
        except Exception as e:
            logger.error(f"Error searching W&B runs: {str(e)}")
            return []


class NoopTracker(ExperimentTracker):
    """
    Dummy tracker that doesn't do anything. 
    Honestly I mostly added this so users can 
    run everything without installing mlflow/wandb.
    Good for quick testing too!
    - HC 5/29/23
    """
    
    def __init__(self, **kwargs):
        """Initialize no-op tracker."""
        self.available = True
        self.active = False
        self.run_id = None
    
    def initialize(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        """Initialize with the given experiment name (no-op)."""
        self.active = True
        return
    
    def start_run(self, run_name: Optional[str] = None, run_id: Optional[str] = None,
                tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new tracking run (no-op)."""
        self.run_id = run_id or f"noop-run-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.active = True
        return self.run_id
    
    def end_run(self) -> None:
        """End the current tracking run (no-op)."""
        self.active = False
        return
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for the current run (no-op)."""
        return
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for the current run (no-op)."""
        return
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log an artifact for the current run (no-op)."""
        return
    
    def log_figure(self, figure: plt.Figure, artifact_name: str) -> None:
        """Log a matplotlib figure for the current run (no-op)."""
        return
    
    def log_model(self, model: torch.nn.Module, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a PyTorch model for the current run (no-op)."""
        return
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], name: str = "confusion_matrix") -> None:
        """Log a confusion matrix for the current run (no-op)."""
        return
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current run (no-op)."""
        return
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get URL to the dashboard for the current run (no-op)."""
        return None
    
    def compare_runs(self, run_ids: List[str], metric_names: List[str], 
                   output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
        """Create a comparison visualization for multiple runs (no-op)."""
        return None
    
    def search_runs(self, filter_string: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for runs matching the filter criteria (no-op)."""
        return []


class ExperimentDashboard:
    """
    One-stop dashboard for tracking experiments.
    
    Realized I needed a way to easily compare experiments
    across different runs without having to remember the
    exact URLs for each system. This is way better!
    """
    
    def __init__(self, tracker: ExperimentTracker):
        """
        Set up the dashboard interface.
        
        Args:
            tracker: The experiment tracker to use
        """
        self.tracker = tracker
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent runs.
        
        Args:
            limit: Max runs to return
            
        Returns:
            List of run info
        """
        return self.tracker.search_runs("", max_results=limit)
    
    def compare_metrics(self, runs: List[str], metrics: List[str], 
                      output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
        """
        Make a chart comparing metrics across different runs.
        
        Really useful for those progress meetings where
        everyone wants to see how the models compare!
        
        Args:
            runs: Run IDs to compare
            metrics: Metrics to put on the chart
            output_path: Where to save the image
            
        Returns:
            The matplotlib figure
        """
        return self.tracker.compare_runs(runs, metrics, output_path)
    
    def filter_runs_by_tags(self, tags: Dict[str, str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find runs that match certain tags.
        
        Args:
            tags: Tags to filter by (name:value)
            limit: Max results to return
            
        Returns:
            Matching runs
        """
        filter_string = " AND ".join([f"tags.{k}='{v}'" for k, v in tags.items()])
        return self.tracker.search_runs(filter_string, max_results=limit)
    
    def filter_runs_by_metrics(self, metrics: Dict[str, Tuple[str, float]], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find runs with metrics in a certain range.
        
        Args:
            metrics: Dict with metric names and (operator, value) tuples
                    e.g. {"accuracy": (">", 0.9)}
            limit: Max results to return
            
        Returns:
            Matching runs
        """
        filter_string = " AND ".join([f"metrics.{k}{op}'{v}'" for k, (op, v) in metrics.items()])
        return self.tracker.search_runs(filter_string, max_results=limit)
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """
        Get the full details of a specific run.
        
        Args:
            run_id: ID of the run to look up
            
        Returns:
            Run details as a dictionary
        """
        runs = self.tracker.search_runs(f"run_id='{run_id}'", max_results=1)
        return runs[0] if runs else {}
    
    def get_dashboard_url(self) -> Optional[str]:
        """
        Get URL to view this run in the web UI.
        
        Returns:
            URL to the dashboard
        """
        return self.tracker.get_dashboard_url() 