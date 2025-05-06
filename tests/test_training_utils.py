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

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules to test
from src.training_utils import (
    EarlyStopping,
    get_scheduler,
    SchedulerType,
    apply_gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    plot_lr_schedule
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.layer(x)

class TestTrainingUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment once before all tests."""
        # Create temporary directory for tests
        cls.temp_dir = Path(tempfile.mkdtemp())
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Setup before each test."""
        # Create a simple model and optimizer for testing
        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
    def test_early_stopping_min_mode(self):
        """Test early stopping in 'min' mode."""
        # Create early stopping object
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # Test improvement detection
        self.assertTrue(early_stopping(0.9))  # First value, always improves
        self.assertTrue(early_stopping(0.8))  # Improved
        self.assertFalse(early_stopping(0.85))  # Worsened, counter = 1
        self.assertFalse(early_stopping(0.82))  # Improved but not enough (< min_delta), counter = 2
        self.assertTrue(early_stopping(0.75))  # Improved enough
        self.assertFalse(early_stopping(0.8))  # Worsened, counter = 1
        self.assertFalse(early_stopping(0.85))  # Worsened, counter = 2
        self.assertFalse(early_stopping(0.9))  # Worsened, counter = 3
        
        # Check early stop flag
        self.assertTrue(early_stopping.early_stop)
        
        # Check trace
        self.assertEqual(len(early_stopping.trace), 8)
        self.assertEqual(early_stopping.best_score, 0.75)
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in 'max' mode."""
        # Create early stopping object
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        
        # Test improvement detection
        self.assertTrue(early_stopping(0.7))  # First value, always improves
        self.assertTrue(early_stopping(0.8))  # Improved
        self.assertFalse(early_stopping(0.78))  # Worsened, counter = 1
        self.assertFalse(early_stopping(0.77))  # Worsened, counter = 2
        
        # Check early stop flag
        self.assertTrue(early_stopping.early_stop)
        
        # Check trace and best score
        self.assertEqual(len(early_stopping.trace), 4)
        self.assertEqual(early_stopping.best_score, 0.8)
    
    def test_get_scheduler(self):
        """Test scheduler creation."""
        # Test StepLR scheduler
        scheduler = get_scheduler(SchedulerType.STEP, self.optimizer)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test ExponentialLR scheduler
        scheduler = get_scheduler(SchedulerType.EXPONENTIAL, self.optimizer, gamma=0.9)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        self.assertEqual(scheduler.gamma, 0.9)
        
        # Test CosineAnnealingLR scheduler
        scheduler = get_scheduler(SchedulerType.COSINE, self.optimizer, T_max=100)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 100)
        
        # Test ReduceLROnPlateau scheduler
        scheduler = get_scheduler(SchedulerType.REDUCE_ON_PLATEAU, self.optimizer, 
                                factor=0.5, patience=5)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 5)
        
        # Test OneCycleLR scheduler
        scheduler = get_scheduler(SchedulerType.ONE_CYCLE, self.optimizer, 
                                max_lr=0.1, steps_per_epoch=10, epochs=5)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        
        # Test with string scheduler type
        scheduler = get_scheduler("step", self.optimizer)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test with invalid scheduler type
        with self.assertRaises(ValueError):
            get_scheduler("invalid_type", self.optimizer)
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        # Create dummy input and target
        x = torch.randn(1, 10)
        target = torch.tensor([0])
        
        # Forward pass
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Get parameter with largest gradient before clipping
        before_clip = max([p.grad.abs().max().item() for p in self.model.parameters()])
        
        # Apply clipping with small max_norm
        max_norm = 0.01
        apply_gradient_clipping(self.model, max_norm=max_norm)
        
        # Get parameter with largest gradient after clipping
        after_clip = max([p.grad.abs().max().item() for p in self.model.parameters()])
        
        # Verify gradient is clipped
        self.assertLessEqual(after_clip, max_norm)
        
        # If the gradient was large enough, it should be reduced
        if before_clip > max_norm:
            self.assertLess(after_clip, before_clip)
        
        # Test adaptive clipping
        self.model.zero_grad()
        
        # Create a new computation graph
        new_x = torch.randn(1, 10)
        new_target = torch.tensor([0])
        new_output = self.model(new_x)
        new_loss = nn.CrossEntropyLoss()(new_output, new_target)
        
        # Backward pass on the new computation graph
        new_loss.backward()
        
        # Apply adaptive clipping
        apply_gradient_clipping(self.model, max_norm=1.0, adaptive=True, model_type='arcface')
        
        # Get parameter with largest gradient after adaptive clipping
        after_adaptive = max([p.grad.abs().max().item() for p in self.model.parameters()])
        
        # Verify gradient is clipped according to the model type (arcface has max_norm=0.3)
        self.assertLessEqual(after_adaptive, 0.3)
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading."""
        # Create checkpoint directory
        checkpoint_dir = self.temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create some validation metrics
        val_metrics = {"loss": 0.5, "accuracy": 0.85}
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            epoch=10,
            validation_metrics=val_metrics,
            checkpoint_dir=checkpoint_dir,
            filename="test_checkpoint.pth",
            metadata={"test_metadata": True}
        )
        
        # Verify checkpoint was saved
        self.assertTrue(checkpoint_path.exists())
        
        # Change model parameters
        original_params = [p.clone() for p in self.model.parameters()]
        for p in self.model.parameters():
            nn.init.uniform_(p)
        
        # Verify parameters changed
        for i, p in enumerate(self.model.parameters()):
            self.assertFalse(torch.allclose(p, original_params[i]))
        
        # Load checkpoint
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )
        
        # Verify checkpoint data
        self.assertEqual(checkpoint["epoch"], 10)
        self.assertEqual(checkpoint["validation_metrics"], val_metrics)
        self.assertTrue(checkpoint["metadata"]["test_metadata"])
        
        # Verify model parameters were restored
        for i, p in enumerate(self.model.parameters()):
            self.assertTrue(torch.allclose(p, original_params[i]))
    
    def test_checkpoint_pruning(self):
        """Test checkpoint pruning."""
        # Create checkpoint directory
        checkpoint_dir = self.temp_dir / "prune_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create multiple checkpoints
        for i in range(10):
            # Create dummy file
            with open(checkpoint_dir / f"checkpoint_{i}.pth", 'w') as f:
                f.write("dummy content")
        
        # Verify 10 checkpoints exist
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
        self.assertEqual(len(checkpoints), 10)
        
        # Prune to keep only 3 checkpoints
        prune_checkpoints(checkpoint_dir, keep=3, pattern="checkpoint_*.pth")
        
        # Verify only 3 checkpoints remain
        remaining = list(checkpoint_dir.glob("checkpoint_*.pth"))
        self.assertEqual(len(remaining), 3)
    
    def test_plot_lr_schedule(self):
        """Test learning rate schedule plotting."""
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        
        # Create plot file path
        plot_path = self.temp_dir / "lr_schedule.png"
        
        # Plot schedule
        plot_lr_schedule(scheduler, self.optimizer, num_epochs=20, save_path=plot_path)
        
        # Verify plot was saved
        self.assertTrue(plot_path.exists())
        
        # Verify optimizer LR was reset
        self.assertEqual(self.optimizer.param_groups[0]['lr'], 0.01)

if __name__ == '__main__':
    unittest.main() 