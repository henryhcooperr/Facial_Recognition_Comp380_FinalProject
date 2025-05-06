#!/usr/bin/env python3

import os
import sys
import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import time

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules to test
from src.advanced_metrics import (
    calculate_per_class_metrics,
    rank_classes_by_difficulty,
    create_enhanced_confusion_matrix,
    expected_calibration_error,
    temperature_scaling,
    optimize_temperature,
    TimerContext,
    ResourceMonitor,
    count_model_parameters
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class TestAdvancedMetrics(unittest.TestCase):
    """Test suite for advanced metrics functionality."""
    
    def setUp(self):
        """Set up test data and models."""
        # Create a temp directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create dummy classification data
        self.num_classes = 5
        self.num_samples = 100
        
        # Create ground truth and predictions
        self.y_true = np.random.randint(0, self.num_classes, size=self.num_samples)
        
        # Create predictions with some errors to make it realistic
        self.y_pred = np.copy(self.y_true)
        error_indices = np.random.choice(self.num_samples, size=20, replace=False)
        for idx in error_indices:
            self.y_pred[idx] = (self.y_true[idx] + 1) % self.num_classes
        
        # Create probability scores
        self.y_score = np.random.random((self.num_samples, self.num_classes))
        # Make predicted class have higher probability
        for i in range(self.num_samples):
            self.y_score[i] = self.y_score[i] / self.y_score[i].sum()  # Normalize
            self.y_score[i, self.y_pred[i]] = max(0.5, self.y_score[i, self.y_pred[i]] * 2)
            self.y_score[i] = self.y_score[i] / self.y_score[i].sum()  # Normalize again
        
        # Create class names
        self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        # Create a dummy model
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, self.num_classes)
        )
    
    def tearDown(self):
        """Clean up test resources."""
        # Remove the temp directory
        shutil.rmtree(self.test_dir)
    
    def test_calculate_per_class_metrics(self):
        """Test per-class metrics calculation."""
        # Calculate per-class metrics
        metrics = calculate_per_class_metrics(
            self.y_true, self.y_pred, self.y_score, self.class_names
        )
        
        # Check that we have metrics for each class
        self.assertEqual(len(metrics), self.num_classes)
        
        # Check that each class has the expected metrics
        for class_name, class_metrics in metrics.items():
            self.assertIn("precision", class_metrics)
            self.assertIn("recall", class_metrics)
            self.assertIn("f1", class_metrics)
            self.assertIn("support", class_metrics)
            self.assertIn("accuracy", class_metrics)
            self.assertIn("roc_auc", class_metrics)
            
            # Check that metrics are within expected ranges
            self.assertGreaterEqual(class_metrics["precision"], 0.0)
            self.assertLessEqual(class_metrics["precision"], 1.0)
            self.assertGreaterEqual(class_metrics["recall"], 0.0)
            self.assertLessEqual(class_metrics["recall"], 1.0)
            self.assertGreaterEqual(class_metrics["f1"], 0.0)
            self.assertLessEqual(class_metrics["f1"], 1.0)
            self.assertGreaterEqual(class_metrics["accuracy"], 0.0)
            self.assertLessEqual(class_metrics["accuracy"], 1.0)
            
            # If ROC AUC was calculated, check its range
            if not np.isnan(class_metrics["roc_auc"]):
                self.assertGreaterEqual(class_metrics["roc_auc"], 0.0)
                self.assertLessEqual(class_metrics["roc_auc"], 1.0)
    
    def test_rank_classes_by_difficulty(self):
        """Test ranking classes by difficulty."""
        # First calculate per-class metrics
        metrics = calculate_per_class_metrics(
            self.y_true, self.y_pred, self.y_score, self.class_names
        )
        
        # Test ranking by different metrics
        for metric in ["precision", "recall", "f1", "accuracy"]:
            ranked_classes = rank_classes_by_difficulty(metrics, metric=metric)
            
            # Check that we have the right number of classes
            self.assertEqual(len(ranked_classes), self.num_classes)
            
            # Check that ranking is in ascending order (hardest first)
            self.assertTrue(ranked_classes[0][1] <= ranked_classes[-1][1])
            
            # Verify that the values match those in the metrics dictionary
            for class_name, value in ranked_classes:
                self.assertEqual(value, metrics[class_name][metric])
    
    def test_create_enhanced_confusion_matrix(self):
        """Test enhanced confusion matrix creation."""
        # Create enhanced confusion matrix
        cm_data = create_enhanced_confusion_matrix(self.y_true, self.y_pred, self.class_names)
        
        # Check that it contains the expected components
        self.assertIn("matrix", cm_data)
        self.assertIn("class_names", cm_data)
        self.assertIn("class_statistics", cm_data)
        
        # Check matrix dimensions
        self.assertEqual(len(cm_data["matrix"]), self.num_classes)
        self.assertEqual(len(cm_data["matrix"][0]), self.num_classes)
        
        # Check class names
        self.assertEqual(cm_data["class_names"], self.class_names)
        
        # Check class statistics
        self.assertEqual(len(cm_data["class_statistics"]), self.num_classes)
        
        # Check that each class has the expected statistics
        for class_name, stats in cm_data["class_statistics"].items():
            self.assertIn("true_positives", stats)
            self.assertIn("false_positives", stats)
            self.assertIn("false_negatives", stats)
            self.assertIn("precision", stats)
            self.assertIn("recall", stats)
            self.assertIn("support", stats)
            self.assertIn("misclassified_to", stats)
    
    def test_expected_calibration_error(self):
        """Test expected calibration error calculation."""
        # Calculate ECE
        ece_data = expected_calibration_error(self.y_true, self.y_pred, self.y_score)
        
        # Check that it contains the expected components
        self.assertIn("expected_calibration_error", ece_data)
        self.assertIn("maximum_calibration_error", ece_data)
        self.assertIn("bin_accuracies", ece_data)
        self.assertIn("bin_confidences", ece_data)
        self.assertIn("bin_counts", ece_data)
        self.assertIn("n_bins", ece_data)
        
        # Check ECE is within expected range
        self.assertGreaterEqual(ece_data["expected_calibration_error"], 0.0)
        self.assertLessEqual(ece_data["expected_calibration_error"], 1.0)
        
        # Check MCE is within expected range
        self.assertGreaterEqual(ece_data["maximum_calibration_error"], 0.0)
        self.assertLessEqual(ece_data["maximum_calibration_error"], 1.0)
        
        # Check bin dimensions
        self.assertEqual(len(ece_data["bin_accuracies"]), ece_data["n_bins"])
        self.assertEqual(len(ece_data["bin_confidences"]), ece_data["n_bins"])
        self.assertEqual(len(ece_data["bin_counts"]), ece_data["n_bins"])
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        # Create some logits (before softmax)
        logits = torch.randn(self.num_samples, self.num_classes)
        
        # Test with different temperatures
        for temp in [0.5, 1.0, 2.0]:
            # Apply temperature scaling
            scaled_probs = temperature_scaling(logits, temp)
            
            # Check shape
            self.assertEqual(scaled_probs.shape, logits.shape)
            
            # Check that probabilities sum to 1 for each sample
            for i in range(self.num_samples):
                self.assertAlmostEqual(np.sum(scaled_probs[i]), 1.0, places=6)
            
            # For temp > 1, probabilities should be more uniform
            # For temp < 1, probabilities should be more peaked
            if temp > 1.0:
                # Get max prob for each sample
                max_probs = np.max(scaled_probs, axis=1)
                # Compare with original softmax
                orig_probs = torch.nn.functional.softmax(logits, dim=1).numpy()
                orig_max_probs = np.max(orig_probs, axis=1)
                # Higher temp should give lower max probs (more uniform)
                self.assertTrue(np.mean(max_probs) < np.mean(orig_max_probs))
            elif temp < 1.0:
                # Get max prob for each sample
                max_probs = np.max(scaled_probs, axis=1)
                # Compare with original softmax
                orig_probs = torch.nn.functional.softmax(logits, dim=1).numpy()
                orig_max_probs = np.max(orig_probs, axis=1)
                # Lower temp should give higher max probs (more peaked)
                self.assertTrue(np.mean(max_probs) > np.mean(orig_max_probs))
    
    def test_optimize_temperature(self):
        """Test temperature optimization."""
        # Create logits with different distributions for optimal temperature
        
        # Case 1: Well-calibrated model (optimal T should be reasonable)
        # For simplicity, we'll make this model predict everything correctly
        logits1 = torch.zeros(self.num_samples, self.num_classes)
        for i in range(self.num_samples):
            logits1[i, self.y_true[i]] = 2.0  # Strong correct prediction
        
        # Optimize temperature
        optimal_temp1 = optimize_temperature(logits1, torch.tensor(self.y_true))
        
        # Check that optimal temperature is a positive number (not checking exact value)
        self.assertGreater(optimal_temp1, 0.0)
        
        # Case 2: Overconfident model (optimal T > 1.0)
        logits2 = torch.zeros(self.num_samples, self.num_classes)
        for i in range(self.num_samples):
            logits2[i, self.y_true[i]] = 5.0  # Very strong correct prediction
        # Add some errors to make it overconfident
        error_indices = np.random.choice(self.num_samples, size=20, replace=False)
        for idx in error_indices:
            wrong_class = (self.y_true[idx] + 1) % self.num_classes
            logits2[idx, self.y_true[idx]] = 0.0
            logits2[idx, wrong_class] = 5.0
        
        # Optimize temperature
        optimal_temp2 = optimize_temperature(logits2, torch.tensor(self.y_true))
        
        # For an overconfident model, optimal temperature should be > 1.0
        self.assertGreater(optimal_temp2, 1.0)
    
    def test_timer_context(self):
        """Test timer context manager."""
        # Use timer context
        with TimerContext("Test Operation") as timer:
            # Sleep for a known duration
            time.sleep(0.1)
        
        # Check that timer recorded a duration close to the sleep time
        self.assertGreaterEqual(timer.elapsed_time, 0.09)
        self.assertLessEqual(timer.elapsed_time, 0.2)
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        # Initialize resource monitor
        monitor = ResourceMonitor(log_interval=0.01)
        
        # Start monitoring
        monitor.start()
        
        # Do some work
        for _ in range(10000):
            _ = [i * i for i in range(1000)]
        
        # Stop monitoring
        data = monitor.stop()
        
        # Check that it contains the expected components
        self.assertIn("duration", data)
        self.assertIn("readings_count", data)
        self.assertIn("cpu_percent", data)
        self.assertIn("memory_mb", data)
        
        # Check that we have some readings
        self.assertGreater(data["readings_count"], 0)
        
        # Check CPU percentage is within sensible range
        self.assertGreaterEqual(data["cpu_percent"]["mean"], 0.0)
        self.assertLessEqual(data["cpu_percent"]["mean"], 100.0)
        
        # Check memory usage is positive
        self.assertGreater(data["memory_mb"]["mean"], 0.0)
    
    def test_count_model_parameters(self):
        """Test counting model parameters."""
        # Count parameters
        param_counts = count_model_parameters(self.model)
        
        # Check that it contains the expected components
        self.assertIn("total_parameters", param_counts)
        self.assertIn("trainable_parameters", param_counts)
        self.assertIn("non_trainable_parameters", param_counts)
        
        # For our simple model, all parameters should be trainable
        self.assertEqual(param_counts["total_parameters"], param_counts["trainable_parameters"])
        self.assertEqual(param_counts["non_trainable_parameters"], 0)
        
        # Calculate expected parameters manually
        # First layer: 10 inputs * 20 outputs + 20 biases = 220
        # Second layer: 20 inputs * 5 outputs + 5 biases = 105
        expected_params = 220 + (20 * self.num_classes + self.num_classes)
        self.assertEqual(param_counts["total_parameters"], expected_params)
        
        # Make some parameters non-trainable
        for param in list(self.model.parameters())[:2]:
            param.requires_grad = False
        
        # Count parameters again
        param_counts = count_model_parameters(self.model)
        
        # Check that we now have non-trainable parameters
        self.assertGreater(param_counts["non_trainable_parameters"], 0)
        self.assertEqual(
            param_counts["total_parameters"],
            param_counts["trainable_parameters"] + param_counts["non_trainable_parameters"]
        )


if __name__ == "__main__":
    unittest.main() 