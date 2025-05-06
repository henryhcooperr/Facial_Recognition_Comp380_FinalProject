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

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw images downloaded from datasets
│   └── processed/          # Preprocessed face images
├── outputs/
│   ├── checkpoints/        # Saved models
│   └── visualizations/     # Training curves and visualizations
├── src/
│   ├── base_config.py      # Configuration and paths
│   ├── data_prep.py        # Data preprocessing
│   ├── download_dataset.py # Dataset downloading
│   ├── face_models.py      # Model architectures
│   ├── interactive.py      # Interactive menu
│   ├── main.py             # Command line interface
│   ├── testing.py          # Model evaluation
│   ├── training.py         # Model training
│   └── visualize.py        # Visualization functions
├── tests/                  # Unit tests
├── requirements.txt        # Required packages
└── run.py                  # Main entry point
```

## Development Journey

I started this project with high ambitions but quickly faced reality - deep learning is hard! My journey went something like this:

1. **First attempt**: Just a simple CNN trained from scratch. Worked OK but overfitted quickly.

2. **Learning about transfer learning**: Discovered ResNet18 pretrained models - what a difference! Suddenly my accuracy jumped from 63% to 87%.

3. **Data challenges**: Dealing with imbalanced datasets was a pain. Tried oversampling, undersampling, and finally settled on a combination of data augmentation + class weights.

4. **My first all-nighter**: Spent all night fixing a mysterious bug where validation accuracy would randomly drop to 0%. Turned out to be a subtle issue with my data loader.

5. **Discovering Siamese networks**: After reading up on face recognition literature, I implemented a Siamese network for one-shot learning. This was a game-changer for handling new faces with limited examples.


## License

This project is for educational purposes as part of the COMP 380 Neural Networks class. 
