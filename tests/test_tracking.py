#!/usr/bin/env python3

import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt

from src.tracking import (
    ExperimentTracker,
    MLflowTracker,
    WeightsAndBiasesTracker,
    NoopTracker,
    ExperimentDashboard
)



class MockTracker(ExperimentTracker):
    """
    Fake tracker for testing
    
    I created this instead of messing with complex mocking libraries
    Simple but gets the job done
    """
    
    def __init__(self, **kwargs):
        # Track everything that gets logged
        self.params_log = {}
        self.metrics = {}  # Changed from metrics_log for inconsistency
        self.artifacts_log = []
        self.figures_log = []
        self.models_log = []
        self.confusion_matrices_log = []
        self.tags_log = {}
        self.search_results = []
        self.dashboard_url = "https://mock-dashboard.com/experiment/123"
        self.compare_runs_result = None
        self.initialized = False
        self.active = False
        self.run_id = None
    
    def initialize(self, experiment_name: str, tracking_uri: str = None) -> None:
        # Just store stuff for verification later
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.initialized = True
    
    def start_run(self, run_name: str = None, run_id: str = None, tags: dict = None) -> str:
        self.run_name = run_name
        self.run_id = run_id or "mock-run-id"
        self.run_tags = tags or {}
        self.active = True
        return self.run_id
    
    def end_run(self) -> None:
        self.active = False
    
    def log_params(self, params: dict) -> None:
        # Just store the params
        self.params_log.update(params)
    
    def log_metrics(self, metrics: dict, step: int = None) -> None:
        # This is a bit clunky but works for testing - HC 5/17/23
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append((v, step))
    
    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        self.artifacts_log.append((local_path, artifact_path))
    
    def log_figure(self, figure: plt.Figure, artifact_name: str) -> None:
        self.figures_log.append((figure, artifact_name))
    
    def log_model(self, model, model_name: str, metadata: dict = None) -> None:
        self.models_log.append((model, model_name, metadata))
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: list, name: str = "confusion_matrix") -> None:
        self.confusion_matrices_log.append((cm, class_names, name))
    
    def set_tags(self, tags: dict) -> None:
        self.tags_log.update(tags)
    
    def get_dashboard_url(self) -> str:
        return self.dashboard_url
    
    def compare_runs(self, run_ids: list, metric_names: list, output_path: str = None):
        # Need to track what gets called for verification
        self.compare_runs_called_with = {"run_ids": run_ids, "metric_names": metric_names, "output_path": output_path}
        return self.compare_runs_result
    
    def search_runs(self, filter_string: str, max_results: int = 100) -> list:
        # Just to track what got called
        self.search_runs_called_with = {"filter_string": filter_string, "max_results": max_results}
        return self.search_results


class TestExperimentDashboard(unittest.TestCase):
    """
    Test the ExperimentDashboard class
    
    This was tricky because we need to test with different tracker types
    """
    
    def setUp(self):
        """Set up for each test"""
        # Make a mock tracker we can inspect
        self.mockedTracker = MockTracker()
        self.dashboard = ExperimentDashboard(self.mockedTracker)
        
        # Pretend we have some runs
        self.fakeRuns = [
            {"id": "run1", "name": "Run 1", "metrics": {"accuracy": 0.95, "loss": 0.05}, "tags": {"version": "v1"}},
            {"id": "run2", "name": "Run 2", "metrics": {"accuracy": 0.90, "loss": 0.1}, "tags": {"version": "v2"}},
            {"id": "run3", "name": "Run 3", "metrics": {"accuracy": 0.85, "loss": 0.15}, "tags": {"version": "v1"}}
        ]
        self.mockedTracker.search_results = self.fakeRuns
        
        # Mock figure - note we need to clean this up later
        self.mockFig = plt.figure()
        self.mockedTracker.compare_runs_result = self.mockFig
    
    def tearDown(self):
        """Clean up after each test"""
        # Make sure we don't leak figures
        plt.close(self.mockFig)
    
    def test_initialization(self):
        """Just make sure the dashboard got set up right"""
        self.assertEqual(self.dashboard.tracker, self.mockedTracker)
    
    def test_get_recent_runs(self):
        """Should be able to get recent runs"""
        # Get with a limit
        runs = self.dashboard.get_recent_runs(limit=5)
        
        # The search should be called right
        self.assertEqual(self.mockedTracker.search_runs_called_with["filter_string"], "")
        self.assertEqual(self.mockedTracker.search_runs_called_with["max_results"], 5)
        
        # And we should get our fake runs back
        self.assertEqual(runs, self.fakeRuns)
    
    def test_compare_metrics(self):
        """Test comparing metrics across runs"""
        # runs to compare
        runs = ["run1", "run2"]
        metrics = ["accuracy", "loss"]
        
        # Try it without output path first
        fig = self.dashboard.compare_metrics(runs, metrics)
        
        # Check it called compare_runs right
        self.assertEqual(self.mockedTracker.compare_runs_called_with["run_ids"], runs)
        self.assertEqual(self.mockedTracker.compare_runs_called_with["metric_names"], metrics)
        self.assertIsNone(self.mockedTracker.compare_runs_called_with["output_path"])
        
        # And it returned the figure
        self.assertEqual(fig, self.mockFig)
        
        # Now try with an output path
        # Create a temp dir for the output
        with tempfile.TemporaryDirectory() as tmpDir:
            output_path = Path(tmpDir) / "comparison.png"
            fig = self.dashboard.compare_metrics(runs, metrics, output_path)
            
            # It should pass the output path through
            self.assertEqual(self.mockedTracker.compare_runs_called_with["output_path"], output_path)
    
    def test_filter_runs_by_tags(self):
        """Test filtering by tags"""
        # Set up some tags to filter by
        myTags = {"version": "v1", "model": "resnet"}
        runs = self.dashboard.filter_runs_by_tags(myTags, limit=10)
        
        # It should search with the right filter string
        expected = "tags.version='v1' AND tags.model='resnet'"
        self.assertEqual(self.mockedTracker.search_runs_called_with["filter_string"], expected)
        self.assertEqual(self.mockedTracker.search_runs_called_with["max_results"], 10)
        
        # And return our fake runs
        self.assertEqual(runs, self.fakeRuns)
    
    def test_filter_runs_by_metrics(self):
        """Test filtering by metric thresholds"""
        # Want runs with high accuracy and low loss
        metricFilters = {
            "accuracy": (">", 0.9),
            "loss": ("<", 0.1)
        }
        runs = self.dashboard.filter_runs_by_metrics(metricFilters, limit=15)
        
        # It should build a filter string with all the metric conditions
        expected = "metrics.accuracy>'0.9' AND metrics.loss<'0.1'"
        self.assertEqual(self.mockedTracker.search_runs_called_with["filter_string"], expected)
        self.assertEqual(self.mockedTracker.search_runs_called_with["max_results"], 15)
        
        # And return our fake runs
        self.assertEqual(runs, self.fakeRuns)
    
    def test_get_run_details(self):
        """Test getting details for one run"""
        run_id = "run1"
        details = self.dashboard.get_run_details(run_id)
        
        # Should search for this specific run
        expected = "run_id='run1'"
        self.assertEqual(self.mockedTracker.search_runs_called_with["filter_string"], expected)
        self.assertEqual(self.mockedTracker.search_runs_called_with["max_results"], 1)
        
        # And return the first (only) run from the results
        self.assertEqual(details, self.fakeRuns[0])
    
    def test_get_dashboard_url(self):
        """Test getting the dashboard URL"""
        url = self.dashboard.get_dashboard_url()
        self.assertEqual(url, self.mockedTracker.dashboard_url)


class TestTrackerCreation(unittest.TestCase):
    """
    Tests for creating different trackers with the factory
    
    Added this 5/18 because I kept forgetting the right string names
    """
    
    def test_create_mlflow_tracker(self):
        """Should create an MLflow tracker"""
        tracker = ExperimentTracker.create("mlflow")
        self.assertIsInstance(tracker, MLflowTracker)
    
    def test_create_wandb_tracker(self):
        """Should create a W&B tracker with both name formats"""
        tracker = ExperimentTracker.create("wandb")
        self.assertIsInstance(tracker, WeightsAndBiasesTracker)
        
        # Should work with the long name too
        tracker = ExperimentTracker.create("weights_and_biases")
        self.assertIsInstance(tracker, WeightsAndBiasesTracker)
    
    def test_create_noop_tracker(self):
        """Should create a no-op tracker"""
        tracker = ExperimentTracker.create("none")
        self.assertIsInstance(tracker, NoopTracker)
    
    def test_create_unknown_tracker(self):
        """Should fall back to NoopTracker for unknown types"""
        # FIXME: Should probably raise an exception instead?
        # This would log a warning
        tracker = ExperimentTracker.create("unknown_tracker_type")
        self.assertIsInstance(tracker, NoopTracker)


# Added this 5/20 - need to make sure NoopTracker doesn't blow up
class TestNoopTracker(unittest.TestCase):
    """Tests for the NoopTracker implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = NoopTracker()
    
    def test_initialize(self):
        """Just make sure it sets active=True"""
        self.tracker.initialize("Test Experiment")
        self.assertTrue(self.tracker.active)
    
    def test_start_run(self):
        """Should get a run ID and set active"""
        run_id = self.tracker.start_run("Test Run")
        self.assertTrue(self.tracker.active)
        # ID should start with "noop-run-" followed by a timestamp
        self.assertTrue(run_id.startswith("noop-run-"))
    
    def test_end_run(self):
        """Test ending a run."""
        self.tracker.start_run("Test Run")
        self.tracker.end_run()
        self.assertFalse(self.tracker.active)
    
    def test_get_dashboard_url(self):
        """Should return None (no dashboard)"""
        url = self.tracker.get_dashboard_url()
        self.assertIsNone(url)
    
    def test_noop_methods(self):
        """All these methods should do nothing"""
        # None of these should raise exceptions
        self.tracker.log_params({"param1": 1, "param2": "test"})
        self.tracker.log_metrics({"metric1": 0.5, "metric2": 0.9})
        self.tracker.log_artifact("nonexistent_file.txt")
        self.tracker.log_figure(plt.figure(), "test_figure.png")
        self.tracker.log_model(MagicMock(), "test_model")
        self.tracker.log_confusion_matrix(np.array([[1, 2], [3, 4]]), ["class1", "class2"])
        self.tracker.set_tags({"tag1": "value1"})
        self.assertIsNone(self.tracker.compare_runs(["run1", "run2"], ["metric1"]))
        self.assertEqual(self.tracker.search_runs("filter"), [])


if __name__ == "__main__":
    unittest.main() 