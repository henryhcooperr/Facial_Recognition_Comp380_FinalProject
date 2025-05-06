#!/usr/bin/env python3

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import numpy as np

from src.experiment_manager import ExperimentConfig
from src.tracking import ExperimentTracker, ExperimentDashboard
from src.experiment_manager import ResultsManager

# Had to redo this entire test suite when I added the dashboard feature

class TestTrackingIntegration(unittest.TestCase):
    """
    Tests for how tracking system integrates with the rest of the code
    
    These are more realistic end-to-end tests than the unit tests
    """
    
    def setUp(self):
        """Create test environment with dummy data"""
        # Make a temp directory for test outputs
        self.tmp = tempfile.mkdtemp()
        self.resultsDir = Path(self.tmp)
        
        # Build a simple config that uses NoopTracker
        # Needs all the right fields for tracking to work
        self.cfg = ExperimentConfig(
            experiment_name="Test Experiment",
            results_dir=self.resultsDir,
            tracker_type=ExperimentConfig.TrackerType.NONE,  # Use NoopTracker for tests
            track_metrics=True,
            track_params=True,
            track_artifacts=True,
            track_models=True
        )
    
    def tearDown(self):
        """Clean up the temp directory"""
        # Remove test directory and all files
        shutil.rmtree(self.tmp)
    
    @patch('src.tracking.ExperimentTracker.create')
    def test_results_manager_creates_dashboard(self, mockCreate):
        """Check if ResultsManager creates and initializes a dashboard"""
        # Set up a mock tracker
        mockTracker = MagicMock()
        mockCreate.return_value = mockTracker
        
        # Create ResultsManager with our mocked tracker
        rm = ResultsManager(self.cfg)
        
        # The create function should be called with the right type
        mockCreate.assert_called_once_with("none")  
        mockTracker.initialize.assert_called_once()
        mockTracker.start_run.assert_called_once()
        
        # Should have a dashboard
        self.assertTrue(hasattr(rm, 'dashboard'))
        self.assertIsInstance(rm.dashboard, ExperimentDashboard)
    
    @patch('src.tracking.ExperimentTracker.create')
    def test_dashboard_access_methods(self, mockCreate):
        """Check that we can access the dashboard methods from ResultsManager"""
        # Create a mock tracker that returns specific values
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_url.return_value = "https://example.com/dashboard"
        
        # Make it return some fake runs when searched
        fake_runs = [
            {"id": "run1", "metrics": {"accuracy": 0.95}},
            {"id": "run2", "metrics": {"accuracy": 0.90}}
        ]
        mock_tracker.search_runs.return_value = fake_runs
        
        # And a fake figure for comparison
        fake_fig = MagicMock()
        mock_tracker.compare_runs.return_value = fake_fig
        
        # Use our mock
        mockCreate.return_value = mock_tracker
        
        # Create ResultsManager
        results = ResultsManager(self.cfg)
        
        # Test getting dashboard URL
        url = results.dashboard.get_dashboard_url()
        self.assertEqual(url, "https://example.com/dashboard")
        
        # Test getting recent runs
        runs = results.dashboard.get_recent_runs(limit=5)
        self.assertEqual(runs, fake_runs)
        mock_tracker.search_runs.assert_called_with("", max_results=5)
        
        # Test comparing metrics
        runIds = ["run1", "run2"]
        metricsToCompare = ["accuracy", "loss"]
        fig = results.dashboard.compare_metrics(
            runIds, 
            metricsToCompare
        )
        self.assertEqual(fig, fake_fig)
        mock_tracker.compare_runs.assert_called_with(
            runIds, 
            metricsToCompare, 
            None
        )
    
    # Testing MLflow integration - had to troubleshoot this a lot
    # Problem was MLflow needs the right return value - HC 6/1/23
    @patch('src.tracking.MLflowTracker')
    def test_mlflow_tracking_integration(self, MockMLflowTracker):
        """Test that MLflow tracking works properly"""
        # Set up a config that uses MLflow
        mlflow_cfg = ExperimentConfig(
            experiment_name="MLflow Test",
            results_dir=self.resultsDir,
            tracker_type=ExperimentConfig.TrackerType.MLFLOW,
            tracking_uri="http://localhost:5000",  # Doesn't need to exist for test
            track_metrics=True,
            track_params=True
        )
        
        # Create a mock MLflow tracker
        mock_ml = MagicMock()
        MockMLflowTracker.return_value = mock_ml
        mock_ml.available = True  # Important! This was my bug.
        
        # Create ResultsManager that will use our mock
        with patch('src.tracking.ExperimentTracker.create', return_value=mock_ml):
            rm = ResultsManager(mlflow_cfg)
            
            # Log something and see if it gets through
            rm.record_training_metrics(1, {"loss": 0.5, "accuracy": 0.8})
            
            # Our mock should have received the metrics
            mock_ml.log_metrics.assert_called()
    
    # Also had issues with W&B - not as clean an API as MLflow IMHO
    @patch('src.tracking.WeightsAndBiasesTracker')
    def test_wandb_tracking_integration(self, MockWandbTracker):
        """Test that W&B tracking works properly"""
        # Config for W&B
        wandb_cfg = ExperimentConfig(
            experiment_name="W&B Test",
            results_dir=self.resultsDir,
            tracker_type=ExperimentConfig.TrackerType.WANDB,
            wandb_project="face-recognition-test",
            wandb_entity="test-user",
            track_metrics=True,
            track_params=True
        )
        
        # Create mock W&B tracker
        mock_wandb = MagicMock()
        MockWandbTracker.return_value = mock_wandb
        mock_wandb.available = True
        
        # Create ResultsManager
        with patch('src.tracking.ExperimentTracker.create', return_value=mock_wandb):
            rm = ResultsManager(wandb_cfg)
            
            # Log metrics and verify they were logged
            rm.record_training_metrics(1, {"loss": 0.5, "accuracy": 0.8})
            mock_wandb.log_metrics.assert_called()
    
    def test_end_to_end_workflow(self):
        """
        Test a complete tracking workflow using NoopTracker
        
        This is a happy path test that makes sure the whole system works together
        It doesn't verify every detail but checks the main outputs
        """
        # Create ResultsManager with NoopTracker
        resultsMgr = ResultsManager(self.cfg)
        
        # Track some training progress
        resultsMgr.record_training_metrics(1, {"loss": 0.5, "accuracy": 0.8})
        resultsMgr.record_training_metrics(2, {"loss": 0.4, "accuracy": 0.85})
        
        # Evaluation metrics too
        resultsMgr.record_evaluation_metrics(1, {"val_loss": 0.6, "val_accuracy": 0.75})
        resultsMgr.record_evaluation_metrics(2, {"val_loss": 0.55, "val_accuracy": 0.8})
        
        # Final test results
        resultsMgr.record_test_metrics({"test_accuracy": 0.82, "test_f1": 0.81})
        
        # Confusion matrix data
        # This also tests class-wise metrics logging
        cm = np.array([[10, 2], [1, 8]])  # Just some made-up numbers
        y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        resultsMgr.record_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=["Class 0", "Class 1"]
        )
        
        # Check if all the CSV files were created
        outputDir = self.resultsDir
        self.assertTrue((outputDir / "metrics" / "training_metrics.csv").exists())
        self.assertTrue((outputDir / "metrics" / "evaluation_metrics.csv").exists())
        self.assertTrue((outputDir / "metrics" / "test_metrics.csv").exists())
        self.assertTrue((outputDir / "logs" / "experiment_log.json").exists())
        
        # Generate the summary and check it
        summary = resultsMgr.generate_experiment_summary()
        
        # Verify some key info
        self.assertEqual(summary["experiment_name"], "Test Experiment")
        self.assertTrue("test_metrics" in summary)
        
        # The dashboard URL should be None for NoopTracker
        dashboardUrl = resultsMgr.dashboard.get_dashboard_url()
        self.assertIsNone(dashboardUrl)
    
    # Had a weird issue where model checkpoints weren't being logged correctly
    # Fixed it 6/5 but adding this test just to be sure - HC
    @patch('src.tracking.ExperimentTracker.create')
    def test_model_checkpoint_tracking(self, mockCreate):
        """Test that model checkpoints get tracked"""
        # Create a mock tracker
        mockTracker = MagicMock()
        mockCreate.return_value = mockTracker
        
        # Create ResultsManager
        rm = ResultsManager(self.cfg)
        
        # Create a dummy model to checkpoint
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save a checkpoint and mark it as best
        rm.save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            is_best=True,
            metrics={"accuracy": 0.9, "loss": 0.1}
        )
        
        # Verify the checkpoint files were created
        self.assertTrue((self.resultsDir / "checkpoints" / "checkpoint_epoch_5.pth").exists())
        self.assertTrue((self.resultsDir / "checkpoints" / "best_model.pth").exists())
        
        # If track_models is True, the tracker should log the model
        if self.cfg.track_models:
            mockTracker.log_model.assert_called()


if __name__ == "__main__":
    unittest.main() 