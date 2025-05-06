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
    Gets detailed metrics per class so we can see which classes are hard to classify
    
    y_true: ground truth labels  
    y_pred: what the model predicted
    y_score: confidence scores
    class_names: list of class names
    
    Returns a dict with metrics for each class
    """
    # Get precision, recall, etc. for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # results dict - will store everything here
    results = {}
    
    for i, cls in enumerate(class_names):
        # Skip non-existent classes
        if i >= len(precision):
            continue
            
        # Get ROC AUC - this took me forever to get right ugh
        # Create binary labels for this class
        true_bin = (np.array(y_true) == i).astype(int)
        
        try:
            # Handle different score formats
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                scores_i = y_score[:, i]
            else:
                # For binary cases
                scores_i = y_score if i == 1 else 1 - y_score
            
            # Could do this in one line but breaking it up for readability
            fpr, tpr, _ = roc_curve(true_bin, scores_i)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"ROC AUC failed for {cls}: {str(e)}")
            roc_auc = float('nan')
        
        # Calculate accuracy for this class
        n_right = sum((y_true == i) & (y_pred == i))
        n_total = sum(y_true == i)
        accuracy = n_right / n_total if n_total > 0 else 0
        
        # Store all metrics for this class
        results[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc)
        }
    
    return results


def rank_classes_by_difficulty(per_class_metrics: Dict[str, Dict[str, float]], 
                              metric: str = "f1") -> List[Tuple[str, float]]:
    """
    Sorts classes by how hard they are (based on some metric)
    
    Lower metric value = harder to classify
    """
    # This is a bit ugly but does the job
    classes_with_scores = []
    for cls, metrics in per_class_metrics.items():
        # Skip classes with no score for the metric
        if metric not in metrics:
            continue
        classes_with_scores.append((cls, metrics[metric]))
    
    # Sort by score (ascending)
    # should probably use key= but this way works fine too
    classes_with_scores.sort(key=lambda x: x[1])
    
    return classes_with_scores


def create_enhanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                    class_names: List[str]) -> Dict[str, Any]:
    """
    Makes a beefed-up confusion matrix with extra stats
    
    This is the function I wrote to analyze my Alzheimer's model
    performance back in April - really helpful for understanding what's 
    going wrong with the model predictions.
    """
    # Regular confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate the totals
    rows_total = cm.sum(axis=1)
    cols_total = cm.sum(axis=0)
    
    # Per-class stats
    class_info = {}
    
    for i, cls in enumerate(class_names):
        if i < len(rows_total):
            # TP, FP, FN
            true_positives = cm[i, i]
            false_positives = cols_total[i] - true_positives
            false_negatives = rows_total[i] - true_positives
            
            # Precision, recall
            prec = true_positives / cols_total[i] if cols_total[i] > 0 else 0
            rec = true_positives / rows_total[i] if rows_total[i] > 0 else 0
            
            # Find where this class gets misclassified to
            if rows_total[i] > 0:
                # This is a neat trick to find where misclassifications are going
                # Still not 100% happy with it but works for now
                misclass = []
                for j in range(len(class_names)):
                    if j != i and cm[i, j] > 0:
                        # Calculate % misclassified
                        misclass.append((class_names[j], float(cm[i, j] / rows_total[i])))
                
                # Sort by most common misclassification
                misclass.sort(key=lambda x: x[1], reverse=True)
            else:
                misclass = []
            
            # Store everything
            class_info[cls] = {
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "precision": float(prec),
                "recall": float(rec),
                "support": int(rows_total[i]),
                "misclassified_to": misclass[:3]  # Top 3 misclassifications
            }
    
    return {
        "matrix": cm.tolist(),
        "class_names": class_names,
        "class_statistics": class_info
    }


def plot_advanced_confusion_matrix(cm_data: Dict[str, Any], output_path: Optional[Path] = None,
                                 figsize: Tuple[int, int] = (12, 10)):
    """
    Plots a fancy confusion matrix with extra info
    """
    # Get data from dict
    cm = np.array(cm_data["matrix"])
    class_names = cm_data["class_names"]
    
    # Make a custom colormap - I like blue more than the default
    colors = [(1, 1, 1), (0.0, 0.4, 0.8)]  # White to blue
    cmap = LinearSegmentedColormap.from_list("WhiteToBlue", colors, N=100)
    
    # Normalize matrix for better visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Replace NaNs (for empty rows)
    cm_norm = np.nan_to_num(cm_norm)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot matrix
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.set_title('Advanced Confusion Matrix', fontsize=16)
    plt.colorbar(im, ax=ax)
    
    # Set up axes
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add numbers to cells
    threshold = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Show counts and percentages
            cell_text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, cell_text, ha="center", va="center",
                   color="white" if cm_norm[i, j] > threshold else "black")
    
    # Labels
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    
    # Add class stats as a table - this is a bit hacky but looks good
    if "class_statistics" in cm_data:
        # Sort classes by recall
        sorted_classes = sorted(cm_data["class_statistics"].items(), 
                              key=lambda x: x[1]["recall"], reverse=True)
        
        # Create table data
        table_data = []
        for cls, stats in sorted_classes:
            row = [
                cls, 
                f"{stats['precision']:.3f}",
                f"{stats['recall']:.3f}",
                f"{stats['support']}",
            ]
            # Add top misclass if any
            if stats['misclassified_to']:
                top_error = stats['misclassified_to'][0]
                row.append(f"{top_error[0]} ({top_error[1]:.2f})")
            else:
                row.append("None")
            table_data.append(row)
        
        # Add the table below
        plt.table(
            cellText=table_data,
            colLabels=["Class", "Precision", "Recall", "Support", "Top Misclassification"],
            loc='bottom',
            cellLoc='center',
            bbox=[0.0, -0.5, 1.0, 0.3]
        )
        
        # Make room for table
        plt.subplots_adjust(bottom=0.35)
    
    plt.tight_layout()
    
    # Save or show
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
    Shows which classes are giving the model the most trouble
    
    Super useful for my Alzheimer's project - helped me realize we had
    a data quality issue with the mild dementia cases
    """
    # Get classes ranked by difficulty
    ranked = rank_classes_by_difficulty(per_class_metrics, metric)
    
    # Just grab the hardest ones
    trouble_classes = ranked[:n_classes]
    
    # Set up plot
    plt.figure(figsize=figsize)
    
    # Make some lists for plotting
    class_names = [x[0] for x in trouble_classes]
    scores = [x[1] for x in trouble_classes]
    
    # Horizontal bar chart works best
    bars = plt.barh(class_names, scores)
    
    # Add values at end of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{scores[i]:.3f}", va='center')
    
    # Labels etc
    plt.xlabel(f"{metric.capitalize()} Score")
    plt.ylabel("Class")
    plt.title(f"Most Challenging Classes (by {metric.upper()})")
    
    # Show average line
    avg_score = np.mean([metrics[metric] for metrics in per_class_metrics.values()])
    plt.axvline(x=avg_score, color='r', linestyle='--')
    plt.text(avg_score, -0.5, f"Avg: {avg_score:.3f}", color='r')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()




def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_score: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Calculates the Expected Calibration Error (ECE)
    
    ECE measures how well model confidence matches actual accuracy
    For medical imaging, we need well-calibrated models!
    
    Returns dict with calibration metrics
    """
    # Get confidence scores in right format
    if y_score.ndim > 1:
        # Multi-class - get confidence for predicted class
        confs = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    else:
        # Binary case
        confs = y_score
    
    # Split into bins
    bin_indices = np.digitize(confs, np.linspace(0, 1, n_bins))
    
    # Initialize arrays
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calculate accuracy and confidence per bin
    for i in range(n_bins):
        bin_idx = i + 1  # np.digitize starts at 1
        mask = bin_indices == bin_idx
        
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(y_true[mask] == y_pred[mask])
            bin_confs[i] = np.mean(confs[mask])
            bin_counts[i] = np.sum(mask)
    
    # Calculate ECE
    # Tried a few different formulas but this one from the paper works best
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / len(y_true)))
    
    # Maximum Calibration Error (MCE) - worst bin
    mce = np.max(np.abs(bin_accs - bin_confs))
    
    # Return everything 
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "bin_accuracies": bin_accs.tolist(), # needed for plotting
        "bin_confidences": bin_confs.tolist(),
        "bin_counts": bin_counts.tolist(),
        "n_bins": n_bins
    }


def plot_reliability_diagram(calibration_data: Dict[str, Any], 
                           output_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    Makes a reliability diagram to show model calibration
    
    This is the standard way to visualize calibration in papers
    """
    # Get calibration metrics from data dict
    bin_accs = np.array(calibration_data["bin_accuracies"])
    bin_confs = np.array(calibration_data["bin_confidences"])
    bin_counts = np.array(calibration_data["bin_counts"])
    n_bins = calibration_data["n_bins"]
    ece = calibration_data["expected_calibration_error"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Calculate bin centers
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot the actual reliability diagram
    ax.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.3, edgecolor='black', 
          label=f'Expected Calibration Error = {ece:.3f}')
    
    # Show gaps between ideal and actual
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.plot([bin_centers[i], bin_centers[i]], [bin_accs[i], bin_confs[i]], 'r-')
            ax.plot(bin_centers[i], bin_confs[i], 'ro')
    
    # Plot confidence
    ax.plot(bin_centers, bin_confs, 'ro-', label='Average Confidence')
    
    # Add a second y-axis for counts
    # This is a bit fancy but helps interpret the data
    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.1, edgecolor='blue')
    ax2.set_ylabel('Sample Count', color='blue')
    ax2.tick_params(axis='y', colors='blue')
    
    # Set labels
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram (Confidence Calibration)')
    ax.legend(loc='lower right')
    
    # Set limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save or display
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
    Shows distributions of confidence scores for right vs wrong predictions
    
    Helps understand if the model is overly confident in wrong predictions
    """
    # Get confidence in right format
    if y_score.ndim > 1:
        # For multi-class case
        confs = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    else:
        # Binary case
        confs = y_score
    
    # Split into correct/incorrect predictions
    is_correct = y_true == y_pred
    conf_right = confs[is_correct]
    conf_wrong = confs[~is_correct]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot two histograms
    plt.hist(conf_right, bins=n_bins, alpha=0.5, color='green', 
            range=(0, 1), label='Correct Predictions')
    plt.hist(conf_wrong, bins=n_bins, alpha=0.5, color='red', 
            range=(0, 1), label='Incorrect Predictions')
    
    # Labels etc
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution for Correct vs. Incorrect Predictions')
    plt.legend()
    
    # Add stats text at bottom
    avg_conf_right = np.mean(conf_right) if len(conf_right) > 0 else 0
    avg_conf_wrong = np.mean(conf_wrong) if len(conf_wrong) > 0 else 0
    
    stats = (
        f"Correct predictions (n={len(conf_right)}): avg conf = {avg_conf_right:.3f}\n"
        f"Incorrect predictions (n={len(conf_wrong)}): avg conf = {avg_conf_wrong:.3f}\n"
        f"Overall accuracy: {np.mean(is_correct):.3f}"
    )
    plt.figtext(0.5, 0.01, stats, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for stats
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


# This is experimental - tried to implement temperature scaling
# from the paper "On Calibration of Modern Neural Networks"
# Works but still trying to tune it better for Alzheimer's data
def temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling to calibrate confidence scores
    
    T > 1 makes probabilities less extreme (more uncertainty)
    T < 1 makes probabilities more extreme (more certainty)
    
    My rule of thumb: start with T=1.5 and adjust from there
    """
    # Make sure we have a tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        
    # Apply temperature scaling formula from paper
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    return probs.detach().cpu().numpy()



def optimize_temperature(val_logits: np.ndarray, val_true: np.ndarray) -> float:
    """
    Find the optimal temperature for calibration using validation data
    
    Uses LBFGS optimization to minimize NLL loss
    """
    # Convert inputs to torch tensors if needed
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
    
    # Honestly not sure if this loss function is the best choice
    # but it's what they used in the paper
    nll = nn.CrossEntropyLoss()
    
    # Start with T=1 which is no scaling
    temp = nn.Parameter(torch.ones(1) * 1.0)
    
    # Simple optimization loop - this sometimes fails to converge
    # but usually works OK
    opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)
    
    def eval():
        opt.zero_grad()
        scaled = logits / temp
        loss = nll(scaled, labels)
        loss.backward()
        return loss
    
    opt.step(eval)
    
    return temp.item()




class TimerContext:
    """
    Simple timer you can use with 'with' statements
    
    Example:
        with TimerContext("Training") as timer:
            # training code here
        print(f"Training took {timer.elapsed_time:.2f}s")
    """
    
    def __init__(self, name: str):
        """Initialize with a name for the timer."""
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
    Monitors CPU and memory usage during model training/inference
    
    I added this when we kept running out of memory during training -
    helps track down memory leaks and performance issues
    """
    
    def __init__(self, log_interval: int = 5):
        """
        Set up the monitor
        
        Args:
            log_interval: How often to check resource usage (seconds)
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
        
        # Start in separate thread so it doesn't slow down the main code
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring and return the stats
        
        Returns a dict with resource usage statistics
        """
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            # Don't wait too long for thread to exit
            self.monitor_thread.join(timeout=1.0)
            
        # Nothing to report if no readings
        if not self.readings:
            return {"error": "No resource data collected"}
        
        # Organize the data we collected
        cpu = [r["cpu_percent"] for r in self.readings]
        mem = [r["memory_used"] for r in self.readings]
        
        return {
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.start_time,
            "readings_count": len(self.readings),
            "cpu_percent": {
                "mean": np.mean(cpu),
                "max": np.max(cpu),
                "min": np.min(cpu)
            },
            "memory_mb": {
                "mean": np.mean(mem) / (1024 * 1024),  # bytes to MB
                "max": np.max(mem) / (1024 * 1024),
                "min": np.min(mem) / (1024 * 1024)
            },
            "readings": self.readings
        }
    
    def _monitor(self):
        """Internal monitoring loop - don't call directly."""
        # Get current process
        proc = psutil.Process(os.getpid())
        
        # Keep running until told to stop
        while self.monitoring:
            # Get CPU and memory usage
            try:
                # Get CPU percent - sometimes this fails
                cpu = psutil.cpu_percent(interval=0.1)
                mem_info = proc.memory_info()
                
                # Record data point
                self.readings.append({
                    "timestamp": time.time() - self.start_time,
                    "cpu_percent": cpu,
                    "memory_used": mem_info.rss,  # Resident Set Size (actual memory used)
                })
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
            
            # Wait a bit before next reading
            time.sleep(self.log_interval)


# Not fully tested but useful for model comparison
def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the parameters in a model
    """
    # Total params (trainable + frozen)
    total = sum(p.numel() for p in model.parameters())
    
    # Just trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable
    }


# FIXME: This is still experimental and needs work
# Not accurate for all model types yet
# TODO: Make this handle attention models better
def estimate_model_flops(model: torch.nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Rough estimate of FLOPs for a model
    
    Very approximate! Just gives ballpark numbers.
    """
    try:
        # Try to use ptflops if available
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
        # Fallback to manual calculation
        logger.warning("ptflops package not found. Using rough FLOP estimation.")
        
        # Create a dummy input for the model
        dummy_input = torch.randn(input_size)
        
        # Track FLOPs with hooks
        hooks = []
        flops = {"count": 0}
        
        # Define hooks for different layer types
        def conv_hook(module, input, output):
            batch_size = input[0].size(0)
            in_ch = module.in_channels
            out_ch = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            out_size = output.size(2) * output.size(3)
            
            # FLOPs for convolution = 2 * B * C_in * C_out * K * K * H_out * W_out / groups
            ops = 2 * batch_size * in_ch * out_ch * kernel_size * out_size / module.groups
            flops["count"] += ops
        
        def linear_hook(module, input, output):
            batch_size = input[0].size(0)
            # FLOPs for linear = 2 * B * C_in * C_out
            ops = 2 * batch_size * module.in_features * module.out_features
            flops["count"] += ops
        
        # Register hooks on modules we know how to measure
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                hooks.append(mod.register_forward_hook(conv_hook))
            elif isinstance(mod, torch.nn.Linear):
                hooks.append(mod.register_forward_hook(linear_hook))
        
        # Run a forward pass
        with torch.no_grad():
            model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {
            "flops": int(flops["count"]),
            "parameters": count_model_parameters(model)["total_parameters"],
            "source": "simplified_estimate"
        }


def plot_resource_usage(resource_data: Dict[str, Any], 
                       output_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (12, 8)):
    """
    Plot resource usage (CPU, memory) over time
    
    Helps visualize bottlenecks during training
    """
    # Extract data
    times = [r["timestamp"] for r in resource_data["readings"]]
    cpu = [r["cpu_percent"] for r in resource_data["readings"]]
    mem = [r["memory_used"] / (1024 * 1024) for r in resource_data["readings"]]  # to MB
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot CPU
    ax1.plot(times, cpu, 'b-', label='CPU Usage')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('Resource Usage Over Time')
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    
    # Average CPU line
    avg_cpu = resource_data["cpu_percent"]["mean"]
    ax1.axhline(y=avg_cpu, color='r', linestyle='--', label=f'Avg: {avg_cpu:.1f}%')
    ax1.legend()
    
    # Plot memory
    ax2.plot(times, mem, 'g-', label='Memory Usage')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.grid(True)
    
    # Average memory line
    avg_mem = resource_data["memory_mb"]["mean"]
    ax2.axhline(y=avg_mem, color='r', linestyle='--', label=f'Avg: {avg_mem:.1f} MB')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


# This is still WIP - will add more to this later
# -HC 6/15/23
def compare_model_efficiency(models_data: Dict[str, Dict[str, Any]], 
                           output_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Creates a chart comparing efficiency of different models
    
    Shows tradeoffs between accuracy, speed, and model size
    """
    # Get model names
    models = list(models_data.keys())
    
    # Pull out metrics we care about
    inf_times = [data.get('inference_time', 0) for data in models_data.values()]
    flops = [data.get('flops', 0) / 1e9 for data in models_data.values()]  # to GFLOPs
    params = [data.get('parameters', 0) / 1e6 for data in models_data.values()]  # to M params
    accs = [data.get('accuracy', 0) for data in models_data.values()]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot inference time
    axes[0, 0].bar(models, inf_times)
    axes[0, 0].set_ylabel('Inference Time (ms)')
    axes[0, 0].set_title('Inference Time')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot FLOPs
    axes[0, 1].bar(models, flops)
    axes[0, 1].set_ylabel('GFLOPs')
    axes[0, 1].set_title('Computational Complexity')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot parameters
    axes[1, 0].bar(models, params)
    axes[1, 0].set_ylabel('Parameters (M)')
    axes[1, 0].set_title('Model Size')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot accuracy
    axes[1, 1].bar(models, accs)
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Model Performance')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show() 