# Alzheimer's Assistant: Face Recognition System

This project was developed for the COMP 380 Neural Networks class to help Alzheimer's patients recognize family members and close friends through face recognition technology.

## Features

- **Multiple Face Recognition Models**: 
  - Baseline CNN (my first working version)
  - Transfer learning with ResNet18 (most reliable so far)
  - Siamese networks (worked well with limited data)
  - Attention mechanisms (experimental, seems promising)
  - ArcFace (added after reading paper, good for hard examples)
  - Hybrid CNN-Transformer (still debugging this one)

- **Automatic Dataset Management**: Downloads and organizes celebrity and face recognition datasets (spent way too much time on this part!)
- **Interactive Menu**: Easy-to-use interface for all functions (makes demo easier)
- **Data Preprocessing**: Face detection, alignment, and data augmentation
- **Model Training & Evaluation**: Customizable parameters
- **Hyperparameter Tuning**: Uses Optuna to find optimal settings
- **Comprehensive Experiment Tracking**: MLflow and Weights & Biases integration to monitor experiments

## Installation

1. Clone this repository:
```
git clone https://github.com/your-username/alzheimers-assistant.git
cd alzheimers-assistant
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Quick Start

The easiest way to use the system is through the interactive menu:

```
python run.py interactive
```

This will guide you through:
1. Downloading face datasets
2. Processing the data
3. Training different models
4. Evaluating model performance

## Available Datasets

The system can automatically download the following datasets:

1. **Celebrity Faces Dataset**: 18 celebrities with ~100 images each
2. **Face Recognition Dataset**: Multiple subjects with varying numbers of images
3. **LFW Dataset**: Huge dataset, but takes a while to download

When you first run the data processing, the system will automatically download these datasets if they are not already present.

## Command Line Usage

The system can also be used from the command line:

```
# Process raw data
python run.py preprocess

# Train a model (CNN is fastest but ResNet is more accurate)
python run.py train --model-type cnn --batch-size 32 --epochs 50

# Evaluate a model
python run.py evaluate --model-type cnn

# Tune hyperparameters (warning: takes a long time!)
python run.py tune --model-type cnn --n-trials 50

# Check GPU availability
python run.py check-gpu

# List trained models
python run.py list-models

# Run the unit tests
python -m unittest discover -s tests
```

## Experiment Configuration and Testing

### YAML Configuration

The system now supports YAML-based configuration for experiments, making it easier to reproduce and share experiment settings. You can define all experiment parameters in a YAML file and run the experiment using:

```
python run.py experiment --config my_experiment.yaml
```

Example YAML configuration file:

```yaml
experiment_name: "CNN with Data Augmentation"
dataset: "both"
model_architecture: "cnn"
preprocessing_config:
  name: "augmented"
  use_mtcnn: true
  face_margin: 0.4
  final_size: [224, 224]
  augmentation: true
epochs: 30
batch_size: 32
learning_rate: 0.001
cross_dataset_testing: true
tracker_type: "mlflow"  # Options: "mlflow", "wandb", "none"
track_metrics: true
track_params: true
track_artifacts: true
track_models: true
```

You can also generate a template configuration:

```
python run.py generate-config --output my_experiment.yaml
```

### Architecture Comparison

You can compare multiple architectures in a single experiment by specifying them as a list:

```yaml
experiment_name: "Architecture Comparison"
dataset: "both"
model_architecture: ["baseline", "cnn", "attention", "arcface"]
epochs: 30
batch_size: 32
learning_rate: 0.001
```

### Testing Framework

The system includes comprehensive tests to ensure all components work correctly:

```
# Run all tests
python -m unittest discover -s tests

# Run specific tests for the experiment manager
python tests/run_experiment_tests.py

# Run specific test file
python -m unittest tests/test_face_recognition_system.py
```

The test suite covers:
- YAML configuration management
- Model registry operations
- Results management
- Different experiment types
- Cross-dataset testing
- Architecture comparison

## Experiment Tracking

The system includes comprehensive experiment tracking integration with MLflow and Weights & Biases.

### MLflow Tracking

MLflow is used by default to track experiment metrics, parameters, artifacts, and models:

```
# Run an experiment with MLflow tracking (default)
python run.py experiment --config my_experiment.yaml

# Run an experiment with specific tracking system
python run.py experiment --config my_experiment.yaml --track mlflow

# Run an experiment without tracking
python run.py experiment --config my_experiment.yaml --no-track
```

To view MLflow dashboard and experiment results:

```
# View the MLflow dashboard
python run.py dashboard --type mlflow

# Compare specific runs
python run.py dashboard --type mlflow --compare run1_id run2_id --metrics accuracy precision recall
```

### Weights & Biases Integration

The system also supports Weights & Biases for experiment tracking:

```
# Run with W&B tracking
python run.py experiment --config my_experiment.yaml --track wandb
```

To configure W&B, set your entity and project in `base_config.py` or environment variables:

```python
# in base_config.py
WANDB_PROJECT = "face-recognition"
WANDB_ENTITY = "your-username"  # or team name
```

To view W&B dashboard and experiment results:

```
# Get the W&B dashboard URL
python run.py dashboard --type wandb

# List experiments tracked with W&B
python run.py list-models --runs --tracker wandb
```

### Tracking Features

The experiment tracking system includes:

- **Metric Tracking**: Automatically logs training, validation, and test metrics
- **Parameter Tracking**: Records all experiment configuration parameters
- **Artifact Logging**: Saves visualizations, confusion matrices, and results
- **Model Registry**: Logs and versions trained models
- **Experiment Comparison**: Compare metrics across different experiment runs
- **Run History**: Maintains searchable history of all experiment runs

### Unified Dashboard Interface

The system provides a unified dashboard interface to interact with tracking systems:

```python
from src.tracking import ExperimentTracker, ExperimentDashboard

# Create tracker
tracker = ExperimentTracker.create("mlflow")
tracker.initialize("My Experiment")

# Create dashboard interface
dashboard = ExperimentDashboard(tracker)

# Get recent runs
runs = dashboard.get_recent_runs(limit=5)

# Compare metrics across runs
dashboard.compare_metrics(
    ["run_id1", "run_id2"], 
    ["accuracy", "precision", "recall"]
)

# Filter runs by metrics
dashboard.filter_runs_by_metrics({
    "accuracy": (">", 0.9),
    "loss": ("<", 0.1)
})
```

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw images downloaded from datasets
│   └── processed/          # Preprocessed face images
├── outputs/
│   ├── checkpoints/        # Saved models
│   ├── visualizations/     # Training curves and visualizations
│   └── tracking/           # Experiment tracking data
│       ├── mlflow/         # MLflow tracking files
│       └── wandb/          # W&B tracking files
├── src/
│   ├── base_config.py      # Configuration and paths
│   ├── data_prep.py        # Data preprocessing
│   ├── download_dataset.py # Dataset downloading
│   ├── face_models.py      # Model architectures
│   ├── interactive.py      # Interactive menu
│   ├── main.py             # Command line interface
│   ├── testing.py          # Model evaluation
│   ├── training.py         # Model training
│   ├── training_utils.py   # Advanced training utilities
│   ├── experiment_manager.py # Experiment configuration and execution
│   ├── tracking.py         # Experiment tracking abstraction
│   ├── advanced_metrics.py # Enhanced evaluation metrics
│   └── visualize.py        # Visualization functions
├── tests/                  # Unit tests
├── requirements.txt        # Required packages
├── run.py                  # Main entry point
└── rerun_experiment.py     # Script for rerunning specific experiments
```

## Experiment Rerun Capabilities

The system now includes a powerful experiment rerun feature that allows you to selectively rerun specific parts of your experiments without starting from scratch. This is especially useful when:

- One model (like Siamese) failed while others succeeded
- You need to rerun cross-validation with different parameters
- You want to try a different hyperparameter optimization approach
- You fixed a bug and only want to rerun affected components

### Using the Rerun Script

The easiest way to use this feature is with the interactive prompt:

```
python rerun_experiment.py
```

This will guide you through:
1. Selecting an existing experiment to rerun
2. Choosing which models to rerun (e.g., "siamese", "arcface", or "all")
3. Deciding whether to rerun cross-validation and hyperparameter optimization
4. Starting fresh or resuming from checkpoints
5. Confirming which directories to delete for clean reruns

### Command Line Options

For automated reruns, you can use the command line arguments:

```
# Rerun just the siamese model with a fresh start (no checkpoint resumption)
python rerun_experiment.py --mode rerun --experiment-id comprehensive_experiment_20250506_182408 --rerun-models siamese --fresh-start

# Rerun multiple models with auto-confirmation (no deletion prompts)
python rerun_experiment.py --mode rerun --experiment-id your_experiment_id --rerun-models cnn attention --auto-confirm

# Rerun cross-validation but keep existing model results
python rerun_experiment.py --mode rerun --experiment-id your_experiment_id --rerun-cv
```

### Rerun Features

The rerun functionality includes several advanced features:

- **Selective Model Rerunning**: Only rerun specific model architectures
- **Cross-Validation Control**: Rerun or skip cross-validation
- **Hyperparameter Optimization Control**: Rerun or skip hyperopt
- **Fresh Start Option**: Disable checkpoint resumption for clean runs
- **Smart Directory Cleanup**: Intelligently finds and removes old data
- **Safe Deletion Handling**: Avoids errors when deleting nested directories
- **Auto-Confirm Mode**: Enables unattended rerunning in scripts

When rerunning, the script maintains the original experiment configuration but allows you to restart specific components from scratch.

## Training Enhancements

The system now includes advanced training enhancements that improve model performance, training stability, and efficiency:

### Early Stopping

Automatically stops training when performance on validation data stops improving, saving time and preventing overfitting:

```yaml
# Early stopping configuration in YAML
use_early_stopping: true
early_stopping_patience: 10  # Number of epochs to wait for improvement
early_stopping_min_delta: 0.001  # Minimum change to count as improvement
early_stopping_metric: "accuracy"  # Metric to monitor
early_stopping_mode: "max"  # Use "max" for accuracy, "min" for loss
```

### Gradient Clipping

Helps prevent exploding gradients, stabilizing training for more complex models:

```yaml
# Gradient clipping configuration in YAML
use_gradient_clipping: true
gradient_clipping_max_norm: 1.0  # Maximum gradient norm
gradient_clipping_adaptive: true  # Automatically adjust based on model type
```

### Learning Rate Scheduling

Multiple learning rate schedulers to optimize training:

```yaml
# Learning rate scheduler configuration in YAML
lr_scheduler_type: "cosine"  # Options: step, exponential, cosine, reduce_on_plateau, one_cycle
lr_scheduler_params:
  T_max: 50  # For cosine scheduler
  eta_min: 0.0001
```

Available schedulers:
- **Step**: Reduces learning rate by a factor at specified steps
- **Exponential**: Decreases learning rate exponentially
- **Cosine**: Follows cosine curve from initial LR to eta_min
- **ReduceOnPlateau**: Reduces LR when a metric stops improving
- **OneCycle**: One-cycle policy as described in Leslie Smith's paper

### Enhanced Model Checkpointing

Improved checkpoint management for better experiment tracking and resumable training:

```yaml
# Checkpoint configuration in YAML
save_best_checkpoint: true  # Save best model as checkpoint
checkpoint_frequency: 5  # Save every N epochs
keep_last_n_checkpoints: 3  # Only keep the last N checkpoints
keep_best_n_checkpoints: 1  # Keep best N checkpoints
save_checkpoint_metadata: true  # Include metadata in checkpoints
resumable_training: true  # Enable resuming from checkpoints
```

### Visualization Improvements

Added learning rate schedule visualization:

```python
from src.training_utils import plot_lr_schedule
plot_lr_schedule(scheduler, optimizer, num_epochs=50, save_path='lr_schedule.png')
```

### Using Training Enhancements

To use these enhancements in an experiment, add the appropriate parameters to your YAML configuration:

```yaml
experiment_name: "Enhanced CNN Training"
dataset: "dataset1"
model_architecture: "cnn"
epochs: 50
batch_size: 32
learning_rate: 0.001
random_seed: 42

# Early stopping
use_early_stopping: true
early_stopping_patience: 7
early_stopping_metric: "accuracy"
early_stopping_mode: "max"

# Gradient clipping
use_gradient_clipping: true
gradient_clipping_max_norm: 1.0

# Learning rate scheduling
lr_scheduler_type: "cosine"
lr_scheduler_params:
  T_max: 50
  eta_min: 0.00001

# Checkpointing
save_best_checkpoint: true
checkpoint_frequency: 1
keep_last_n_checkpoints: 5
```

### Testing Training Enhancements

Run the training enhancement tests to verify everything is working correctly:

```
python tests/run_training_tests.py
```

## Enhanced Evaluation Metrics

The framework provides comprehensive evaluation capabilities to understand model performance in depth:

1. **Per-class performance analysis**:
   - Detailed metrics (precision, recall, F1, ROC-AUC) for each class
   - Enhanced confusion matrices with per-class statistics
   - Class difficulty ranking to identify problematic classes
   - Visualizations highlighting performance differences between classes

2. **Confidence calibration metrics**:
   - Expected Calibration Error (ECE) to measure prediction reliability
   - Reliability diagrams showing confidence vs. accuracy
   - Confidence histograms for correct vs. incorrect predictions
   - Temperature scaling for calibrating model outputs

3. **Time and resource utilization metrics**:
   - Comprehensive timing for training, epochs, and inference
   - Memory usage tracking during model execution
   - Model complexity analysis (parameters, FLOPs)
   - Comparison of efficiency across architectures

These detailed metrics help identify model weaknesses, improve reliability, and optimize resource usage in production deployments.

## Development Journey

I started this project with high ambitions but quickly faced reality - deep learning is hard! My journey went something like this:

1. **First attempt**: Just a simple CNN trained from scratch. Worked OK but overfitted quickly.

2. **Learning about transfer learning**: Discovered ResNet18 pretrained models - what a difference! Suddenly my accuracy jumped from 63% to 87%.

3. **Data challenges**: Dealing with imbalanced datasets was a pain. Tried oversampling, undersampling, and finally settled on a combination of data augmentation + class weights.

4. **My first all-nighter**: Spent all night fixing a mysterious bug where validation accuracy would randomly drop to 0%. Turned out to be a subtle issue with my data loader.

5. **Discovering Siamese networks**: After reading up on face recognition literature, I implemented a Siamese network for one-shot learning. This was a game-changer for handling new faces with limited examples.

## License

This project is for educational purposes as part of the COMP 380 Neural Networks class. 
