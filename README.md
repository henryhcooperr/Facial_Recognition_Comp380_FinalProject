# Alzheimer's Assistant: Face Recognition System

This project was developed for the COMP 380 Neural Networks class to help Alzheimer's patients recognize family members and close friends through face recognition technology.

## Personal Motivation

After watching my grandmother struggle with Alzheimer's and repeatedly not recognizing family members, I wanted to create a system that could help patients maintain connections with their loved ones. This system identifies faces in images and provides the names of people the patient knows, offering a bridge between memory loss and personal relationships.

## Features

- **Multiple Face Recognition Models**: Baseline CNN, Transfer learning with ResNet18, Siamese networks, Attention mechanisms, ArcFace, and hybrid CNN-Transformer architectures
- **Automatic Dataset Management**: Downloads and organizes celebrity and face recognition datasets
- **Interactive Menu**: Easy-to-use interface for all functions
- **Data Preprocessing**: Includes face detection, alignment, and data augmentation
- **Model Training & Evaluation**: Train models with customizable parameters and evaluate their performance
- **Hyperparameter Tuning**: Automatically find optimal model configurations

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

When you first run the data processing, the system will automatically download these datasets if they are not already present.

## Command Line Usage

The system can also be used from the command line:

```
# Process raw data
python run.py preprocess

# Train a model
python run.py train --model-type cnn --batch-size 32 --epochs 50

# Evaluate a model
python run.py evaluate --model-type cnn

# Tune hyperparameters
python run.py tune --model-type cnn --n-trials 50

# Check GPU availability
python run.py check-gpu

# List trained models
python run.py list-models
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

I started with a simple CNN classifier, but as I learned more about neural networks, I expanded the project to include more sophisticated architectures. I spent significant time experimenting with hyperparameters, data augmentation techniques, and different model architectures to improve recognition accuracy.

## Future Improvements

- Mobile app integration for real-time recognition
- Emotion detection to provide context about the person's mood
- Voice synthesis to speak the person's name and relationship
- Memory assistance by providing relevant personal facts about the recognized person

## License

This project is for educational purposes as part of the COMP 380 Neural Networks class. 