#!/usr/bin/env python3

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_manager import ExperimentManager, ExperimentConfig
from src.face_models import get_model, BaselineNet
from src.base_config import set_random_seeds

class TestCrossDatasetCompatibility(unittest.TestCase):
    """Test the cross-dataset compatibility fixes."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seeds for reproducibility
        set_random_seeds(42)
        self.source_classes = 31  # Example: dataset1 has 31 classes
        self.target_classes = 18  # Example: dataset2 has 18 classes
    
    def test_model_loading_with_mismatched_classes(self):
        """Test that loading a model with mismatched class count correctly fails."""
        # Create a model with the source dataset class count
        source_model = get_model("baseline", num_classes=self.source_classes)
        
        # Check source model output layer shape
        self.assertEqual(source_model.fc2.weight.shape, torch.Size([self.source_classes, 512]))
        
        # Create another model with target dataset class count
        target_model = get_model("baseline", num_classes=self.target_classes)
        
        # Check target model output layer shape
        self.assertEqual(target_model.fc2.weight.shape, torch.Size([self.target_classes, 512]))
        
        # Save source model weights to a temporary file
        temp_path = "temp_model_test.pth"
        torch.save(source_model.state_dict(), temp_path)
        
        # Try to load source weights into target model - should fail
        with self.assertRaises(RuntimeError):
            target_model.load_state_dict(torch.load(temp_path))
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_model_loading_with_matched_classes(self):
        """Test that loading a model with matched class count works correctly."""
        # Create a model with the source dataset class count
        source_model = get_model("baseline", num_classes=self.source_classes)
        
        # Create another model with the SAME class count
        target_model = get_model("baseline", num_classes=self.source_classes)
        
        # Save source model weights to a temporary file
        temp_path = "temp_model_test.pth"
        torch.save(source_model.state_dict(), temp_path)
        
        # Load source weights into target model - should succeed
        try:
            target_model.load_state_dict(torch.load(temp_path))
            success = True
        except RuntimeError:
            success = False
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Check that loading succeeded
        self.assertTrue(success, "Loading weights with matched class count should succeed")
    
    def test_out_of_bounds_prediction_filtering(self):
        """Test the filtering of out-of-bounds predictions."""
        # Create synthetic predictions and labels
        all_preds = np.array([0, 5, 10, 15, 20, 25, 30])  # Some predictions are out of bounds for target
        all_labels = np.array([0, 1, 2, 3, 4, 5, 6])
        target_class_names = [f"class_{i}" for i in range(self.target_classes)]  # 18 classes
        
        # Filter predictions
        valid_indices = []
        for i, pred in enumerate(all_preds):
            if isinstance(pred, (int, np.integer)) and int(pred) < len(target_class_names):
                valid_indices.append(i)
        
        # Check that indices 0-3 are valid (predictions 0, 5, 10, 15 are within range of 18 classes)
        self.assertEqual(valid_indices, [0, 1, 2, 3])
        
        # Create filtered predictions and labels
        filtered_preds = [all_preds[i] for i in valid_indices]
        filtered_labels = [all_labels[i] for i in valid_indices]
        
        # Check filtered arrays
        np.testing.assert_array_equal(filtered_preds, [0, 5, 10, 15])
        np.testing.assert_array_equal(filtered_labels, [0, 1, 2, 3])

if __name__ == '__main__':
    unittest.main() 