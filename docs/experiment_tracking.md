# Experiment Tracking System

This document explains how to use the experiment tracking system I built for the face recognition project. This was added to help organize runs, compare results, and visualize model performance.

## Why I Built This

I got tired of manually tracking experiments in spreadsheets and screenshots, so I built this tracking system to automate the process. The main goals were:

- Keep track of hyperparameters for each experiment run
- Log metrics like accuracy, loss, etc. during training
- Save artifacts (plots, model checkpoints, etc.)
- Compare different experiments easily
- Have a nice dashboard to view everything

The system supports two tracking backends - MLflow and Weights & Biases. You can also run without any tracking using the "noop" tracker.

## Quick Start

### 1. Choose your tracking system

Set the tracker type in your experiment config:

```python
config = ExperimentConfig(
    # ...other params...
    tracker_type=ExperimentConfig.TrackerType.MLFLOW,  # or WANDB or NONE
    tracking_uri="http://localhost:5000",  # only for MLflow
    wandb_project="face-recognition",      # only for W&B
    wandb_entity="your-username",          # only for W&B
    # ...other params...
)
```

Or set via environment variables:
```bash
export TRACKER_TYPE=mlflow  # or wandb or none
export MLFLOW_TRACKING_URI=http://localhost:5000
export WANDB_PROJECT=face-recognition
export WANDB_ENTITY=your-username
```

### 2. Run your experiment

Just use the normal command:

```bash
python run.py experiment --config my_config.yaml
```

The tracking system will automatically log:
- All hyperparameters
- Training metrics (loss, accuracy, etc.)
- Evaluation metrics
- Test metrics
- Confusion matrices and other visualizations
- Model checkpoints

### 3. View the Dashboard

#### Using MLflow:
```bash
mlflow ui --port 5000
```
Then visit: http://localhost:5000

#### Using W&B:
Visit: https://wandb.ai/your-username/face-recognition

## Using the Dashboard API

If you want to programmatically access the tracking data, you can use the `ExperimentDashboard` class:

```python
from tracking import ExperimentTracker, ExperimentDashboard

# Create a tracker
tracker = ExperimentTracker.create("mlflow", tracking_uri="http://localhost:5000")
tracker.initialize(experiment_name="face-recognition")

# Create dashboard
dashboard = ExperimentDashboard(tracker)

# Get recent runs
recent_runs = dashboard.get_recent_runs(limit=5)

# Compare metrics across runs
dashboard.compare_metrics(
    runs=["run1", "run2", "run3"],
    metrics=["accuracy", "precision", "recall"],
    output_path="comparison.png"
)

# Filter runs by tags
my_runs = dashboard.filter_runs_by_tags(
    tags={"model": "resnet", "dataset": "dataset1"}
)

# Filter runs by metric thresholds
good_runs = dashboard.filter_runs_by_metrics(
    metrics={"accuracy": (">", 0.9), "loss": ("<", 0.1)}
)
```

## Comparing Results

To compare results between runs via CLI:

```bash
python run.py compare-runs --runs run1 run2 run3 --metrics accuracy loss f1_score
```

This will generate a comparison chart and print a summary.

## Tips & Tricks

1. **Always use the same run name format** - I use "MODEL_DATASET_DATE" (e.g., "resnet_dataset1_20230515")

2. **Add tags to runs** - They make filtering much easier:
   ```python
   config.tracking_tags = {
       "model": "resnet",
       "dataset": "dataset1",
       "version": "v1.2"
   }
   ```

3. **MLflow vs W&B** - I found MLflow easier to set up locally, but W&B has nicer visualizations. W&B is also better for sharing results with others.

4. **Naming parameters consistently** - When comparing runs, the dashboard will group by parameter name, so be consistent!

5. **Don't track everything** - Set these to False if you want to save space:
   ```python
   config.track_artifacts = False  # Don't save plots
   config.track_models = False     # Don't save models
   ```

## Known Issues

- Sometimes MLflow loses connection during long runs - I fixed this by setting longer timeouts
- Artifact logging can take a lot of disk space - be selective about what you track
- The dashboard comparison feature only works for metrics logged at the same steps

## Future Improvements

- [ ] Add experiment tagging from CLI
- [ ] Support automatic model registry
- [ ] Better visualization of training curves
- [ ] Direct comparison of model predictions

Let me know if you have any questions or suggestions!

-Henry 