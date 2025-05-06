#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import psutil
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .base_config import logger

def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_score: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate detailed performance metrics for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction confidence scores
        class_names: Names of the classes
        
    Returns:
        Dictionary of per-class metrics
    """
    # Calculate precision, recall, f1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Create per-class metrics dictionary
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Skip if there are no samples for this class
        if i >= len(precision):
            continue
            
        # Calculate ROC curve and AUC for this class
        class_y_true = (np.array(y_true) == i).astype(int)
        
        try:
            # For multi-class, get the scores for the current class
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                class_scores = y_score[:, i]
            else:
                # For binary classification
                class_scores = y_score if i == 1 else 1 - y_score
            
            fpr, tpr, _ = roc_curve(class_y_true, class_scores)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC for class {class_name}: {str(e)}")
            roc_auc = float('nan')
        
        # Calculate accuracy for this class
        class_correct = sum((y_true == i) & (y_pred == i))
        class_total = sum(y_true == i)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        
        # Store metrics for this class
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "accuracy": float(class_accuracy),
            "roc_auc": float(roc_auc)
        }
    
    return per_class_metrics


def rank_classes_by_difficulty(per_class_metrics: Dict[str, Dict[str, float]], 
                              metric: str = "f1") -> List[Tuple[str, float]]:
    """
    Rank classes by difficulty based on a specified metric.
    
    Args:
        per_class_metrics: Dictionary of per-class metrics
        metric: Metric to use for ranking (lower values = more difficult)
        
    Returns:
        List of (class_name, metric_value) tuples sorted by difficulty (hardest first)
    """
    # Create list of (class_name, metric_value) tuples
    class_scores = [(class_name, metrics[metric]) 
                   for class_name, metrics in per_class_metrics.items()]
    
    # Sort by metric value (ascending, so hardest classes first)
    class_scores.sort(key=lambda x: x[1])
    
    return class_scores


def create_enhanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                    class_names: List[str]) -> Dict[str, Any]:
    """
    Create an enhanced confusion matrix with per-class statistics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of the classes
        
    Returns:
        Dictionary with confusion matrix and per-class statistics
    """
    # Calculate regular confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row and column sums
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    
    # Calculate per-class statistics
    class_stats = {}
    
    for i, class_name in enumerate(class_names):
        if i < len(row_sums):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = col_sums[i] - tp
            fn = row_sums[i] - tp
            
            # Precision, recall, specificity
            precision = tp / col_sums[i] if col_sums[i] > 0 else 0
            recall = tp / row_sums[i] if row_sums[i] > 0 else 0
            
            # Misclassification targets (where this class gets misclassified to)
            if row_sums[i] > 0:
                misclassified_to = [(class_names[j], float(cm[i, j] / row_sums[i])) 
                                   for j in range(len(class_names)) if j != i and cm[i, j] > 0]
                misclassified_to.sort(key=lambda x: x[1], reverse=True)
            else:
                misclassified_to = []
            
            # Store statistics
            class_stats[class_name] = {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "support": int(row_sums[i]),
                "misclassified_to": misclassified_to[:3]  # Top 3 misclassifications
            }
    
    return {
        "matrix": cm.tolist(),
        "class_names": class_names,
        "class_statistics": class_stats
    }


def plot_advanced_confusion_matrix(cm_data: Dict[str, Any], output_path: Optional[Path] = None,
                                 figsize: Tuple[int, int] = (12, 10)):
    """
    Create an advanced confusion matrix visualization with per-class statistics.
    
    Args:
        cm_data: Dictionary with confusion matrix data
        output_path: Path to save the visualization
        figsize: Figure size
    """
    cm = np.array(cm_data["matrix"])
    class_names = cm_data["class_names"]
    
    # Create a custom colormap from white to blue
    colors = [(1, 1, 1), (0.0, 0.4, 0.8)]  # White to blue
    cmap = LinearSegmentedColormap.from_list("WhiteToBlue", colors, N=100)
    
    # Normalize confusion matrix for visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.set_title('Advanced Confusion Matrix', fontsize=16)
    plt.colorbar(im, ax=ax)
    
    # Set up axes
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black")
    
    # Add axis labels
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    
    # Add class-level statistics as a table at the bottom
    if "class_statistics" in cm_data:
        # Sort classes by recall (in descending order)
        sorted_classes = sorted(cm_data["class_statistics"].items(), 
                              key=lambda x: x[1]["recall"], reverse=True)
        
        # Create table data
        table_data = []
        for class_name, stats in sorted_classes:
            row = [
                class_name, 
                f"{stats['precision']:.3f}",
                f"{stats['recall']:.3f}",
                f"{stats['support']}",
            ]
            if stats['misclassified_to']:
                top_misclass = stats['misclassified_to'][0]
                row.append(f"{top_misclass[0]} ({top_misclass[1]:.2f})")
            else:
                row.append("None")
            table_data.append(row)
        
        # Create table
        plt.table(
            cellText=table_data,
            colLabels=["Class", "Precision", "Recall", "Support", "Top Misclassification"],
            loc='bottom',
            cellLoc='center',
            bbox=[0.0, -0.5, 1.0, 0.3]
        )
        
        # Adjust layout to make room for the table
        plt.subplots_adjust(bottom=0.35)
    
    plt.tight_layout()
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_class_difficulty_analysis(per_class_metrics: Dict[str, Dict[str, float]], 
                                 output_path: Optional[Path] = None,
                                 metric: str = "f1",
                                 n_classes: int = 10,
                                 figsize: Tuple[int, int] = (12, 8)):
    """
    Create a visualization of the most difficult classes.
    
    Args:
        per_class_metrics: Dictionary of per-class metrics
        output_path: Path to save the visualization
        metric: Metric to use for ranking (lower values = more difficult)
        n_classes: Number of classes to show
        figsize: Figure size
    """
    # Rank classes by difficulty
    ranked_classes = rank_classes_by_difficulty(per_class_metrics, metric)
    
    # Take the n most difficult classes
    most_difficult = ranked_classes[:n_classes]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot metric values for most difficult classes
    class_names = [c[0] for c in most_difficult]
    metric_values = [c[1] for c in most_difficult]
    
    # Create horizontal bar chart
    bars = plt.barh(class_names, metric_values)
    
    # Add data labels to the right of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{metric_values[i]:.3f}", va='center')
    
    # Set labels and title
    plt.xlabel(f"{metric.capitalize()} Score")
    plt.ylabel("Class")
    plt.title(f"Most Challenging Classes (by {metric.upper()})")
    
    # Add a red line for the average score
    avg_score = np.mean([metrics[metric] for metrics in per_class_metrics.values()])
    plt.axvline(x=avg_score, color='r', linestyle='--')
    plt.text(avg_score, -0.5, f"Avg: {avg_score:.3f}", color='r')
    
    plt.tight_layout()
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


# ============================
# Confidence Calibration Metrics
# ============================

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_score: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction confidence scores
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with ECE and related metrics
    """
    if y_score.ndim > 1:
        # For multi-class, get the confidence for the predicted class
        confidences = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    else:
        # For binary classification
        confidences = y_score
    
    # Create bins and compute ECE
    bin_indices = np.digitize(confidences, np.linspace(0, 1, n_bins))
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_idx = i + 1  # np.digitize starts bin indices at 1
        mask = bin_indices == bin_idx
        
        if np.sum(mask) > 0:
            bin_accuracies[i] = np.mean(y_true[mask] == y_pred[mask])
            bin_confidences[i] = np.mean(confidences[mask])
            bin_counts[i] = np.sum(mask)
    
    # Calculate ECE
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * (bin_counts / len(y_true)))
    
    # Calculate Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_accuracies - bin_confidences))
    
    # Return all calibration metrics
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
        "n_bins": n_bins
    }


def plot_reliability_diagram(calibration_data: Dict[str, Any], 
                           output_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    Create a reliability diagram visualization.
    
    Args:
        calibration_data: Dictionary with calibration data
        output_path: Path to save the visualization
        figsize: Figure size
    """
    bin_accuracies = np.array(calibration_data["bin_accuracies"])
    bin_confidences = np.array(calibration_data["bin_confidences"])
    bin_counts = np.array(calibration_data["bin_counts"])
    n_bins = calibration_data["n_bins"]
    ece = calibration_data["expected_calibration_error"]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Set bin edges for the histogram
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot the reliability diagram
    ax.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.3, edgecolor='black', 
          label=f'Expected Calibration Error = {ece:.3f}')
    
    # Plot the gap between confidence and accuracy
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.plot([bin_centers[i], bin_centers[i]], [bin_accuracies[i], bin_confidences[i]], 'r-')
            ax.plot(bin_centers[i], bin_confidences[i], 'ro')
    
    # Plot the average confidence for each bin
    ax.plot(bin_centers, bin_confidences, 'ro-', label='Average Confidence')
    
    # Create a twin y-axis for sample counts
    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.1, edgecolor='blue')
    ax2.set_ylabel('Sample Count', color='blue')
    ax2.tick_params(axis='y', colors='blue')
    
    # Set labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram (Confidence Calibration)')
    ax.legend(loc='lower right')
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_confidence_histogram(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_score: np.ndarray, 
                            output_path: Optional[Path] = None,
                            n_bins: int = 20,
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Create a confidence histogram visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction confidence scores
        output_path: Path to save the visualization
        n_bins: Number of bins for histogram
        figsize: Figure size
    """
    if y_score.ndim > 1:
        # For multi-class, get the confidence for the predicted class
        confidences = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    else:
        # For binary classification
        confidences = y_score
    
    # Separate confidences for correct and incorrect predictions
    correct = y_true == y_pred
    conf_correct = confidences[correct]
    conf_incorrect = confidences[~correct]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot histograms
    plt.hist(conf_correct, bins=n_bins, alpha=0.5, color='green', 
            range=(0, 1), label='Correct Predictions')
    plt.hist(conf_incorrect, bins=n_bins, alpha=0.5, color='red', 
            range=(0, 1), label='Incorrect Predictions')
    
    # Set labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution for Correct vs. Incorrect Predictions')
    plt.legend()
    
    # Add statistics as text
    avg_conf_correct = np.mean(conf_correct) if len(conf_correct) > 0 else 0
    avg_conf_incorrect = np.mean(conf_incorrect) if len(conf_incorrect) > 0 else 0
    
    stats_text = (
        f"Correct predictions (n={len(conf_correct)}): avg conf = {avg_conf_correct:.3f}\n"
        f"Incorrect predictions (n={len(conf_incorrect)}): avg conf = {avg_conf_incorrect:.3f}\n"
        f"Overall accuracy: {np.mean(correct):.3f}"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for stats text
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to the logits to improve calibration.
    
    Args:
        logits: Raw logits from model (before softmax)
        temperature: Temperature parameter (T > 1 smooths probabilities, T < 1 sharpens them)
        
    Returns:
        Calibrated probabilities
    """
    # Convert to tensor if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        
    # Scale logits
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    return probabilities.detach().cpu().numpy()


def optimize_temperature(val_logits: np.ndarray, val_true: np.ndarray) -> float:
    """
    Find the optimal temperature for calibration.
    
    Args:
        val_logits: Validation set logits
        val_true: Validation set ground truth labels
        
    Returns:
        Optimal temperature
    """
    # Convert to torch tensors
    if isinstance(val_logits, np.ndarray):
        logits = torch.from_numpy(val_logits).float()
    elif isinstance(val_logits, torch.Tensor):
        logits = val_logits.clone().detach().float()
    else:
        raise TypeError("val_logits must be numpy array or torch tensor")
        
    if isinstance(val_true, np.ndarray):
        labels = torch.from_numpy(val_true).long()
    elif isinstance(val_true, torch.Tensor):
        labels = val_true.clone().detach().long()
    else:
        raise TypeError("val_true must be numpy array or torch tensor")
    
    # Define temperature scaling as an optimization problem
    nll_criterion = nn.CrossEntropyLoss()
    
    # Initialize temperature to 1
    temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    # Simple optimization loop
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    # Return optimal temperature
    return temperature.item()


# ============================
# Time and Resource Utilization Metrics
# ============================

class TimerContext:
    """
    Context manager for timing operations.
    
    Example:
        with TimerContext("Training epoch") as timer:
            # Do some work
            pass
        print(f"Time taken: {timer.elapsed_time:.2f}s")
    """
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {self.elapsed_time:.2f}s")


class ResourceMonitor:
    """
    Monitor system resource usage during model training/inference.
    """
    
    def __init__(self, log_interval: int = 5):
        """
        Initialize resource monitor.
        
        Args:
            log_interval: How often to log resource usage (in seconds)
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.readings = []
        self.start_time = None
    
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.readings = []
        self.start_time = time.time()
        
        # Start monitoring thread
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring resources.
        
        Returns:
            Dictionary with resource usage statistics
        """
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
        # Calculate statistics
        if not self.readings:
            return {"error": "No readings collected"}
        
        cpu_usage = [r["cpu_percent"] for r in self.readings]
        memory_usage = [r["memory_used"] for r in self.readings]
        
        return {
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.start_time,
            "readings_count": len(self.readings),
            "cpu_percent": {
                "mean": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "min": np.min(cpu_usage)
            },
            "memory_mb": {
                "mean": np.mean(memory_usage) / (1024 * 1024),
                "max": np.max(memory_usage) / (1024 * 1024),
                "min": np.min(memory_usage) / (1024 * 1024)
            },
            "readings": self.readings
        }
    
    def _monitor(self):
        """Internal monitoring loop."""
        process = psutil.Process(os.getpid())
        
        while self.monitoring:
            # Collect CPU and memory usage
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                
                self.readings.append({
                    "timestamp": time.time() - self.start_time,
                    "cpu_percent": cpu_percent,
                    "memory_used": memory_info.rss,  # Resident Set Size in bytes
                })
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
            
            # Sleep for the log interval
            time.sleep(self.log_interval)


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def estimate_model_flops(model: torch.nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Estimate the number of FLOPs in a PyTorch model.
    
    This is a simplified estimate based on common operations. For more accurate
    measurements, use tools like thop or ptflops.
    
    Args:
        model: PyTorch model
        input_size: Input size for the model (batch_size, channels, height, width)
        
    Returns:
        Estimated number of FLOPs
    """
    try:
        # Try to use ptflops package if available
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size[1:], as_strings=False, print_per_layer_stat=False
        )
        return {
            "flops": int(macs * 2),  # FLOPs ≈ 2 * MACs
            "parameters": int(params),
            "source": "ptflops"
        }
    except ImportError:
        # Fall back to a simplified estimate
        logger.warning("ptflops package not found. Using simplified FLOP estimation.")
        
        # Create a dummy input for the model
        dummy_input = torch.randn(input_size)
        
        # Dictionary to store hooks
        hooks = []
        flops_count = {"count": 0}
        
        # Define hook functions for different layer types
        def conv_hook(module, input, output):
            batch_size = input[0].size(0)
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            output_size = output.size(2) * output.size(3)
            
            # FLOPs for convolution = 2 * B * C_in * C_out * K * K * H_out * W_out / groups
            flops = 2 * batch_size * input_channels * output_channels * kernel_size * output_size / module.groups
            flops_count["count"] += flops
        
        def linear_hook(module, input, output):
            batch_size = input[0].size(0)
            # FLOPs for linear = 2 * B * C_in * C_out
            flops = 2 * batch_size * module.in_features * module.out_features
            flops_count["count"] += flops
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
        
        # Run a forward pass to trigger the hooks
        with torch.no_grad():
            model(dummy_input)
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()
        
        return {
            "flops": int(flops_count["count"]),
            "parameters": count_model_parameters(model)["total_parameters"],
            "source": "simplified_estimate"
        }


def plot_resource_usage(resource_data: Dict[str, Any], 
                       output_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (12, 8)):
    """
    Create a visualization of resource usage.
    
    Args:
        resource_data: Dictionary with resource usage data
        output_path: Path to save the visualization
        figsize: Figure size
    """
    # Extract data
    timestamps = [r["timestamp"] for r in resource_data["readings"]]
    cpu_usage = [r["cpu_percent"] for r in resource_data["readings"]]
    memory_usage = [r["memory_used"] / (1024 * 1024) for r in resource_data["readings"]]  # Convert to MB
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot CPU usage
    ax1.plot(timestamps, cpu_usage, 'b-', label='CPU Usage')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('Resource Usage Over Time')
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    
    # Add horizontal line for average CPU usage
    avg_cpu = resource_data["cpu_percent"]["mean"]
    ax1.axhline(y=avg_cpu, color='r', linestyle='--', label=f'Avg: {avg_cpu:.1f}%')
    ax1.legend()
    
    # Plot memory usage
    ax2.plot(timestamps, memory_usage, 'g-', label='Memory Usage')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.grid(True)
    
    # Add horizontal line for average memory usage
    avg_memory = resource_data["memory_mb"]["mean"]
    ax2.axhline(y=avg_memory, color='r', linestyle='--', label=f'Avg: {avg_memory:.1f} MB')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def compare_model_efficiency(models_data: Dict[str, Dict[str, Any]], 
                           output_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Create a visualization comparing model efficiency.
    
    Args:
        models_data: Dictionary with model data (key: model name, value: metrics)
        output_path: Path to save the visualization
        figsize: Figure size
    """
    # Extract model names
    model_names = list(models_data.keys())
    
    # Extract metrics for comparison
    inference_times = [data.get('inference_time', 0) for data in models_data.values()]
    flops = [data.get('flops', 0) / 1e9 for data in models_data.values()]  # Convert to GFLOPs
    parameters = [data.get('parameters', 0) / 1e6 for data in models_data.values()]  # Convert to M params
    accuracies = [data.get('accuracy', 0) for data in models_data.values()]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot inference time
    axes[0, 0].bar(model_names, inference_times)
    axes[0, 0].set_ylabel('Inference Time (ms)')
    axes[0, 0].set_title('Inference Time')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot FLOPs
    axes[0, 1].bar(model_names, flops)
    axes[0, 1].set_ylabel('GFLOPs')
    axes[0, 1].set_title('Computational Complexity')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot parameters
    axes[1, 0].bar(model_names, parameters)
    axes[1, 0].set_ylabel('Parameters (M)')
    axes[1, 0].set_title('Model Size')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot accuracy
    axes[1, 1].bar(model_names, accuracies)
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Model Performance')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure or show it
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show() 