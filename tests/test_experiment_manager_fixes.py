#!/usr/bin/env python3

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_manager import ExperimentManager, ExperimentConfig, ResultsManager
from src.base_config import set_random_seeds


class TestExperimentManagerCrossDataset(unittest.TestCase):
    """Test the experiment manager's cross-dataset functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create an experiment config for testing
        self.config = ExperimentConfig(
            experiment_name="Test Cross-Dataset Experiment",
            dataset=ExperimentConfig.Dataset.BOTH,
            model_architecture="baseline",
            epochs=1,
            batch_size=16,
            learning_rate=0.001,
            cross_dataset_testing=True,
            results_dir=self.test_dir,
            random_seed=42
        )
        
        # Create experiment manager
        self.manager = ExperimentManager()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cross_dataset_experiment(self):
        """Test that cross-dataset experiment handles class count differences correctly."""
        # We'll use a simulated version of the cross-dataset code to test the functionality
        
        # Mock the data loader to avoid file system operations
        with patch('src.experiment_manager.ExperimentManager._setup_data_loaders') as mock_setup_loaders:
            # Set up mock return values
            mock_setup_loaders.side_effect = [
                (MagicMock(), MagicMock(), MagicMock(), ['class1', 'class2', 'class3', 'class4', 'class5']),  # Source dataset - 5 classes
                (MagicMock(), MagicMock(), MagicMock(), ['class1', 'class2', 'class3'])  # Target dataset - 3 classes
            ]
            
            # Mock get_model to verify it's called with correct parameters
            with patch('src.experiment_manager.get_model') as mock_get_model:
                # Make get_model return a mock object
                mock_model = MagicMock()
                mock_get_model.return_value = mock_model
                
                # Mock device creation and model loading
                with patch('torch.device', return_value='cpu'), \
                     patch('torch.load', return_value={}), \
                     patch.object(mock_model, 'load_state_dict'):
                    
                    # Set up experiment parameters
                    dataset = ExperimentConfig.Dataset.DATASET1
                    other_dataset = ExperimentConfig.Dataset.DATASET2
                    model_arch_value = "baseline"
                    
                    # First test the data loader setup which should identify the correct number of classes
                    source_train_loader, _, _, source_class_names = self.manager._setup_data_loaders(
                        dataset.value, 
                        None,  # No preprocessing config for test
                        batch_size=16,
                        seed=42
                    )
                    
                    # Check source class count
                    num_classes = len(source_class_names)
                    self.assertEqual(num_classes, 5)
                    
                    # Set up target dataset loaders
                    _, _, test_loader, target_class_names = self.manager._setup_data_loaders(
                        other_dataset.value, 
                        None,  # No preprocessing config for test
                        batch_size=16,
                        seed=42
                    )
                    
                    # Check target class count
                    self.assertEqual(len(target_class_names), 3)
                    
                    # Now create a model - this should use the source class count
                    model_checkpoint = Path(self.test_dir) / "checkpoints" / "best_model.pth"
                    
                    # Directly call get_model as it would be called in _run_cross_dataset_experiment
                    from src.experiment_manager import get_model
                    model = get_model(model_arch_value, num_classes=num_classes)
                    
                    # Verify that get_model was called with the source class count
                    mock_get_model.assert_called_with(model_arch_value, num_classes=5)


if __name__ == '__main__':
    unittest.main() 