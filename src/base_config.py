#!/usr/bin/env python3

import os
import logging
from pathlib import Path
import sys
import random
import numpy as np
import torch

# Project paths - changed to use pathlib after fighting with os.path for hours
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROC_DATA_DIR = DATA_DIR / "processed"  # processed data goes here
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"  # shortened this
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
VIZ_DIR = OUT_DIR / "visualizations"  

# Added these after I kept typing the wrong paths - lazy but effective
TRAIN_DIR = PROC_DATA_DIR / "train"
VAL_DIR = PROC_DATA_DIR / "val"
TEST_DIR = PROC_DATA_DIR / "test"

# For kaggle datasets
CELEB_DIR = RAW_DATA_DIR / "dataset2"
LFW_DIR = RAW_DATA_DIR / "lfw"

# Model parameters I've tuned through trial and error
# Smaller batch size helped with my limited dataset
DEFAULT_BATCH_SIZE = 16  # was 32 but got OOM errors on my laptop
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3  # 0.001 seems to work well for most models
IMG_SIZE = 224  # This is what ResNet expects

# Make sure all our directories exist
# This annoyed me to no end when things failed silently
for dir_path in [RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR, 
                CHECKPOINTS_DIR, VIZ_DIR,
                TRAIN_DIR, VAL_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging - super helpful for debugging
# Added timestamp after getting confused about when errors happened
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# FIXME: This is a bit clunky but works for now
# Should eventually replace with a proper CLI confirmation
def get_user_confirmation(prompt: str = "Continue? (y/n): ") -> bool:
    """Simple yes/no prompt to get user confirmation.
    
    I use this to avoid accidentally overwriting data or models.
    """
    while True:
        response = input(prompt).lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")

# Quick utility to check if we have a GPU available
def check_gpu():
    """Print GPU info if available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.info("No GPU available, using CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU")
        return False

# Old validation split function I replaced - keeping for reference
# def split_dataset(data_dir, train=0.7, val=0.2, test=0.1):
#     """Split dataset into train/val/test."""
#     assert abs(train + val + test - 1.0) < 1e-8, "Ratios must sum to 1"
#     # rest of function... 

# First attempt at the function to generate model filename 
# def get_model_filename(model_type, accuracy=None):
#     """Generate filename for model checkpoint."""  
#     import time
#     timestamp = int(time.time())
#     if accuracy:
#         return f"{model_type}_{timestamp}_{accuracy:.4f}.pth"
#     return f"{model_type}_{timestamp}.pth" 

def set_random_seeds(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed for random number generators
        deterministic: If True, makes PyTorch operations deterministic (may impact performance)
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for PyTorch to be deterministic
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    logger.info(f"Random seeds have been set to {seed}")
    
    return seed 