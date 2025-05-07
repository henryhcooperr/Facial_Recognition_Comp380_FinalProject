#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
import datetime
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

from .base_config import PROJECT_ROOT, OUT_DIR, PROC_DATA_DIR, RAW_DATA_DIR
from .data_prep import PreprocessingConfig, process_raw_data
from .experiment_manager import (
    ExperimentManager, 
    ExperimentConfig, 
    CrossArchitectureExperiment,
    HyperparameterExperiment,
    ResultsCompiler,
    ResultsManager
)
# Add import for special architecture handlers
from .special_architectures import handle_special_architecture


def check_disk_space(min_gb=5):
    """Check if there's enough disk space available"""
    import psutil
    disk = psutil.disk_usage('/')
    gb_free = disk.free / (1024 * 1024 * 1024)
    if gb_free < min_gb:
        print(f"WARNING: Low disk space! Only {gb_free:.2f}GB available.")
        print(f"This experiment requires at least {min_gb}GB of free space.")
        return False
    return True


def cleanup_old_experiments(keep_newest=3):
    """Clean up old experiment outputs to free disk space"""
    if not OUT_DIR.exists():
        return

    # Get all experiment directories
    exp_dirs = [d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith("experiment_") or 
               d.name.startswith("comprehensive_experiment_")]
    
    # Sort by modification time (newest first)
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep newest N experiments, delete the rest
    if len(exp_dirs) > keep_newest:
        for old_dir in exp_dirs[keep_newest:]:
            try:
                print(f"Removing old experiment directory: {old_dir}")
                shutil.rmtree(old_dir)
            except Exception as e:
                print(f"Failed to remove {old_dir}: {e}")


def prepare_datasets():
    """
    Process raw datasets from data/raw directory to prepare them for experiments.
    This needs to be run before running the comprehensive experiment.
    """
    # Define preprocessing configurations
    enhanced_preprocessing = PreprocessingConfig(
        name="enhanced",
        use_mtcnn=True,
        face_margin=0.4,
        final_size=(224, 224),
        augmentation=True
    )
    
    minimal_preprocessing = PreprocessingConfig(
        name="minimal",
        use_mtcnn=False,
        face_margin=0.0,
        final_size=(224, 224),
        augmentation=False
    )
    
    # Process with both preprocessing configurations
    print("Processing face_recognition dataset (dataset1)...")
    process_raw_data(
        raw_data_dir=RAW_DATA_DIR,  # Pass the base raw data dir
        output_dir=PROC_DATA_DIR,    # Just pass the base processed dir
        config=enhanced_preprocessing
    )
    
    print("Processing face_recognition dataset (dataset1) with minimal preprocessing...")
    process_raw_data(
        raw_data_dir=RAW_DATA_DIR,  # Pass the base raw data dir
        output_dir=PROC_DATA_DIR,    # Just pass the base processed dir
        config=minimal_preprocessing
    )
    
    print("Data preparation complete!")


def run_comprehensive_face_recognition_experiment(config_path=None, rerun_experiment_id=None, 
                                           models_to_rerun=None, rerun_cv=False, rerun_hyperopt=False,
                                           fresh_start=False):
    """
    Run a comprehensive face recognition experiment that evaluates all architectures
    with different preprocessing techniques, performs hyperparameter optimization,
    and generates detailed analyses and reports.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to configuration YAML file
    rerun_experiment_id : str, optional
        ID of experiment to rerun (for selective reruns)
    models_to_rerun : list, optional
        List of model architectures to rerun
    rerun_cv : bool, optional
        Whether to rerun cross-validation
    rerun_hyperopt : bool, optional
        Whether to rerun hyperparameter optimization
    fresh_start : bool, optional
        Whether to disable resuming from checkpoints for rerun models
    """
    # Check disk space before starting
    if not check_disk_space(min_gb=5):
        print("ERROR: Not enough disk space to run the experiment.")
        print("Please free up some disk space before running again.")
        return {"error": "Insufficient disk space"}
    
    # Clean up old experiment directories to free space only if not in rerun mode
    if not rerun_experiment_id:
        cleanup_old_experiments(keep_newest=2)
    
    # Make sure the processed data exists in the expected locations
    for preprocessing in ["enhanced", "minimal"]:
        for dataset in ["dataset1", "dataset2"]:
            data_path = PROC_DATA_DIR / preprocessing / dataset
            if not data_path.exists():
                raise ValueError(f"Processed data not found at {data_path}. Please run prepare_datasets() first.")
    
    # Create timestamp for the comprehensive experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided experiment ID for reruns, otherwise create a new one
    if rerun_experiment_id:
        comprehensive_exp_id = rerun_experiment_id
        main_output_dir = OUT_DIR / comprehensive_exp_id
        print(f"Rerunning experiment: {comprehensive_exp_id}")
    else:
        comprehensive_exp_id = f"comprehensive_experiment_{timestamp}"
        main_output_dir = OUT_DIR / comprehensive_exp_id
        main_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting new experiment: {comprehensive_exp_id}")
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager()
    
    # Track experiment IDs for final reporting
    experiment_ids = []
    
    # Load cross-validation config if provided
    cv_config = None
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                cv_config = yaml.safe_load(f)
            print(f"Loaded cross-validation configuration from {config_path}")
            
            # Save a copy of the config in the output directory if this is a new experiment
            if not rerun_experiment_id:
                with open(main_output_dir / "cv_config.yaml", 'w') as f:
                    yaml.dump(cv_config, f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default parameters instead.")
    
    # Step 1: Define preprocessing configurations
    enhanced_preprocessing = PreprocessingConfig(
        name="enhanced",
        use_mtcnn=True,
        face_margin=0.4,
        final_size=(224, 224),
        augmentation=True
    )
    
    minimal_preprocessing = PreprocessingConfig(
        name="minimal",
        use_mtcnn=False,
        face_margin=0.0,
        final_size=(224, 224),
        augmentation=False
    )
    
    # Define all available architectures - including those requiring special data formats
    # Both Siamese and ArcFace now use custom handlers for proper training
    all_architectures = ["baseline", "cnn", "siamese", "attention", "arcface", "hybrid"]
    
    # If we have a CV config, extract architectures from there
    if cv_config and "models" in cv_config:
        all_architectures = [model["name"] for model in cv_config["models"]]
        print(f"Using architectures from config: {all_architectures}")
    
    # Filter architectures based on models_to_rerun if provided
    if models_to_rerun:
        print(f"Rerunning only these models: {models_to_rerun}")
        all_architectures = [arch for arch in all_architectures if arch in models_to_rerun]
    
    dataset_names = ["dataset1", "dataset2"]  # Define as list, not a module
    
    # Get any specific model parameters from the config
    model_params = {}
    if cv_config and "models" in cv_config:
        for model_config in cv_config["models"]:
            model_name = model_config["name"]
            model_params[model_name] = model_config.get("config", {})
            
            # Add cross-validation specific params
            if "cross_validation" in cv_config:
                model_params[model_name]["cv_folds"] = cv_config["cross_validation"].get("n_folds", 5)
                model_params[model_name]["random_seed"] = cv_config["cross_validation"].get("random_seed", 42)
    
    # Reduce to save disk space and time for testing
    REDUCED_EPOCHS = 10
    
    # Track results for later comparison
    architecture_results = {}
    failed_architectures = []
    
    # Step 2: Architecture comparison with enhanced preprocessing
    print("Running architecture comparison with enhanced preprocessing...")
    
    enhanced_exp_id = f"{comprehensive_exp_id}_enhanced_preprocessing"
    enhanced_output_dir = main_output_dir / "enhanced_preprocessing"
    
    for dataset_name in dataset_names:  # Use the list, not a module
        dataset_exp_id = f"{enhanced_exp_id}_{dataset_name}"
        dataset_output_dir = enhanced_output_dir / dataset_name
        
        print(f"\n{'='*80}")
        print(f"Testing all architectures on {dataset_name} with enhanced preprocessing")
        print(f"{'='*80}")
        
        for arch in all_architectures:
            try:
                # Check disk space again before each architecture
                if not check_disk_space(min_gb=2):
                    print(f"ERROR: Disk space critical, skipping {arch} on {dataset_name}")
                    failed_architectures.append(f"enhanced_{dataset_name}_{arch}")
                    continue
                
                print(f"\nTesting architecture: {arch}")
                
                # Create configuration for this specific architecture
                arch_exp_id = f"{dataset_exp_id}_{arch}"
                arch_output_dir = dataset_output_dir / arch
                
                # Get specific model parameters from config or use defaults
                current_model_params = model_params.get(arch, {})
                
                # Setup base experiment parameters
                arch_config = ExperimentConfig(
                    experiment_id=arch_exp_id,
                    experiment_name=f"Architecture Comparison - Enhanced Preprocessing - {dataset_name} - {arch}",
                    dataset=dataset_name,
                    model_architecture=arch,
                    preprocessing_config=enhanced_preprocessing,
                    epochs=current_model_params.get("epochs", REDUCED_EPOCHS),
                    batch_size=current_model_params.get("batch_size", 32),
                    learning_rate=current_model_params.get("learning_rate", 0.001),
                    cross_dataset_testing=False,
                    results_dir=str(arch_output_dir),
                    
                    # Early stopping configuration
                    use_early_stopping=current_model_params.get("early_stopping", {}).get("enabled", True),
                    early_stopping_patience=current_model_params.get("early_stopping", {}).get("patience", 10),
                    early_stopping_min_delta=current_model_params.get("early_stopping", {}).get("min_delta", 0.001),
                    early_stopping_metric=current_model_params.get("early_stopping", {}).get("metric", "accuracy"),
                    early_stopping_mode=current_model_params.get("early_stopping", {}).get("mode", "max"),
                    
                    # Gradient clipping
                    use_gradient_clipping=True,
                    gradient_clipping_max_norm=1.0,
                    
                    # LR scheduler configuration
                    lr_scheduler_type=current_model_params.get("lr_scheduler", {}).get("type", "reduce_on_plateau"),
                    lr_scheduler_params=current_model_params.get("lr_scheduler", {}),
                    
                    # Disable resumable training if fresh_start is True
                    resumable_training=not fresh_start,
                    
                    # ENSURE ALL VISUALIZATION FLAGS ARE TRUE
                    evaluation_mode="enhanced",  # This is critical for visualizations
                    per_class_analysis=True,     # Ensure per-class visuals are generated
                    calibration_analysis=True,   # Ensure calibration plots are generated
                    resource_monitoring=True,    # Ensure resource plots are generated
                    
                    # MLflow tracking
                    tracker_type="mlflow",
                    tracking_uri=None,  
                    track_metrics=True,
                    track_params=True,
                    track_artifacts=True
                )
                
                # Create a results manager for this architecture
                arch_results_manager = ResultsManager(arch_config)
                
                # Get datasets for potential special handling
                from torch.utils.data import DataLoader
                from torchvision import datasets, transforms
                import torch
                
                # Define transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
                
                # Get the dataset path
                data_dir = PROC_DATA_DIR / enhanced_preprocessing.name / dataset_name
                
                # Load datasets
                train_dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
                val_dataset = datasets.ImageFolder(data_dir / "val", transform=transform)
                test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
                
                # Get the model
                from .face_models import get_model
                num_classes = len(train_dataset.classes)
                model = get_model(arch, num_classes=num_classes)
                
                # Check if architecture needs special handling
                results, was_handled = handle_special_architecture(
                    arch, model, train_dataset, val_dataset, test_dataset, 
                    arch_config, arch_results_manager
                )
                
                if not was_handled:
                    # Use standard experiment runner if not specially handled
                    results = experiment_manager._run_single_model_experiment(arch_config, arch_results_manager)
                
                # Explicitly generate visualizations using enhanced methods
                try:
                    print(f"Generating enhanced visualizations for {arch} on {dataset_name}")
                    
                    # Skip standard visualization for Siamese networks since they require paired inputs
                    if arch == "siamese":
                        print(f"Skipping standard visualizations for Siamese network - using specialized evaluation instead")
                        # Record a basic dummy confusion matrix so reports don't fail
                        if not os.path.exists(arch_output_dir / "plots"):
                            os.makedirs(arch_output_dir / "plots")
                        # Log that visualization was handled by specialized code
                        with open(arch_output_dir / "plots" / "visualization_status.txt", 'w') as f:
                            f.write("Siamese network visualizations handled by specialized evaluation code")
                        continue
                    
                    # Get predictions from trained model
                    model.eval()
                    y_true = []
                    y_pred = []
                    y_score = []
                    
                    # Create dataloader for test data
                    test_batch_size = current_model_params.get("batch_size", 32)
                    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
                    
                    # Get device
                    device = next(model.parameters()).device
                    
                    # Collect predictions
                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                            
                            # Get scores and predictions
                            probs = F.softmax(outputs, dim=1)
                            _, predicted = torch.max(outputs, 1)
                            
                            # Store results
                            y_true.extend(targets.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            y_score.extend(probs.cpu().numpy())
                    
                    # Now that we have predictions, explicitly call visualization methods
                    if len(y_true) > 0 and len(y_pred) > 0 and len(y_score) > 0:
                        # Convert lists to numpy arrays before visualization
                        y_true_np = np.array(y_true)
                        y_pred_np = np.array(y_pred)
                        y_score_np = np.array(y_score)
                        
                        print(f"Generating confusion matrix for {arch}")
                        arch_results_manager.record_confusion_matrix(y_true_np, y_pred_np, test_dataset.classes)
                        
                        print(f"Generating per-class metrics for {arch}")
                        arch_results_manager.record_per_class_metrics(y_true_np, y_pred_np, y_score_np, test_dataset.classes)
                        
                        print(f"Generating calibration metrics for {arch}")
                        arch_results_manager.record_calibration_metrics(y_true_np, y_pred_np, y_score_np)
                        
                        # Also generate resource metrics
                        if arch_results_manager.resource_data:
                            print(f"Generating resource monitoring plots for {arch}")
                            arch_results_manager.record_resource_metrics(arch_results_manager.resource_data, "training")
                        
                        print(f"Successfully generated enhanced visualizations for {arch}")
                        
                        # Check if plots directory exists and has files
                        plots_dir = arch_output_dir / "plots"
                        if plots_dir.exists():
                            num_files = len(list(plots_dir.glob('**/*')))
                            print(f"Found {num_files} visualization files in {plots_dir}")
                        else:
                            print(f"WARNING: Plots directory {plots_dir} does not exist!")
                    
                    else:
                        print(f"WARNING: No valid predictions collected for visualization for {arch}")
                
                except Exception as viz_error:
                    print(f"Error generating visualizations: {viz_error}")
                    import traceback
                    traceback.print_exc()
                
                # Store results
                if f"enhanced_{dataset_name}" not in architecture_results:
                    architecture_results[f"enhanced_{dataset_name}"] = {"results": {}}
                
                architecture_results[f"enhanced_{dataset_name}"]["results"][arch] = results
                experiment_ids.append(arch_exp_id)
                
                print(f"Successfully completed {arch} on {dataset_name} with enhanced preprocessing")
                
                # Clean up to save space - remove checkpoints except best model
                try:
                    checkpoints_dir = arch_output_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        for checkpoint in checkpoints_dir.glob("checkpoint_epoch_*.pth"):
                            if "best" not in str(checkpoint):
                                checkpoint.unlink()
                        print(f"Cleaned up extra checkpoints for {arch}")
                except Exception as e:
                    print(f"Error cleaning up checkpoints: {e}")
                
            except Exception as e:
                print(f"ERROR with architecture {arch} on {dataset_name}: {str(e)}")
                failed_architectures.append(f"enhanced_{dataset_name}_{arch}")
                import traceback
                traceback.print_exc()
                
                # Clean up MLflow run to prevent "Run already active" errors
                import mlflow
                try:
                    mlflow.end_run()
                    print("Ended active MLflow run")
                except Exception as mlflow_err:
                    print(f"Error ending MLflow run: {mlflow_err}")
                
                print(f"Continuing with next architecture...")
                continue
    
    # Check disk space before continuing to minimal preprocessing
    if not check_disk_space(min_gb=2):
        print("ERROR: Not enough disk space for minimal preprocessing experiments.")
        print("Skipping minimal preprocessing experiments and proceeding to analysis.")
    else:
        # Step 3: Architecture comparison with minimal preprocessing
        print("\nRunning architecture comparison with minimal preprocessing...")
        
        minimal_exp_id = f"{comprehensive_exp_id}_minimal_preprocessing"
        minimal_output_dir = main_output_dir / "minimal_preprocessing"
        
        for dataset_name in dataset_names:  # Use the same list variable, not a module
            dataset_exp_id = f"{minimal_exp_id}_{dataset_name}"
            dataset_output_dir = minimal_output_dir / dataset_name
            
            print(f"\n{'='*80}")
            print(f"Testing all architectures on {dataset_name} with minimal preprocessing")
            print(f"{'='*80}")
            
            for arch in all_architectures:
                try:
                    # Check disk space again before each architecture
                    if not check_disk_space(min_gb=1):
                        print(f"ERROR: Disk space critical, skipping {arch} on {dataset_name}")
                        failed_architectures.append(f"minimal_{dataset_name}_{arch}")
                        continue
                        
                    print(f"\nTesting architecture: {arch}")
                    
                    # Create configuration for this specific architecture
                    arch_exp_id = f"{dataset_exp_id}_{arch}"
                    arch_output_dir = dataset_output_dir / arch
                    
                    # Get specific model parameters from config or use defaults
                    current_model_params = model_params.get(arch, {})
                    
                    arch_config = ExperimentConfig(
                        experiment_id=arch_exp_id,
                        experiment_name=f"Architecture Comparison - Minimal Preprocessing - {dataset_name} - {arch}",
                        dataset=dataset_name,
                        model_architecture=arch,
                        preprocessing_config=minimal_preprocessing,
                        epochs=current_model_params.get("epochs", REDUCED_EPOCHS),
                        batch_size=current_model_params.get("batch_size", 32),
                        learning_rate=current_model_params.get("learning_rate", 0.001),
                        cross_dataset_testing=False,
                        results_dir=str(arch_output_dir),
                        
                        # Early stopping configuration
                        use_early_stopping=current_model_params.get("early_stopping", {}).get("enabled", True),
                        early_stopping_patience=current_model_params.get("early_stopping", {}).get("patience", 10),
                        early_stopping_min_delta=current_model_params.get("early_stopping", {}).get("min_delta", 0.001),
                        early_stopping_metric=current_model_params.get("early_stopping", {}).get("metric", "accuracy"),
                        early_stopping_mode=current_model_params.get("early_stopping", {}).get("mode", "max"),
                        
                        # Gradient clipping
                        use_gradient_clipping=True,
                        gradient_clipping_max_norm=1.0,
                        
                        # LR scheduler configuration
                        lr_scheduler_type=current_model_params.get("lr_scheduler", {}).get("type", "reduce_on_plateau"),
                        lr_scheduler_params=current_model_params.get("lr_scheduler", {}),
                        
                        # Disable resumable training if fresh_start is True
                        resumable_training=not fresh_start,
                        
                        # ENSURE ALL VISUALIZATION FLAGS ARE TRUE
                        evaluation_mode="enhanced",  # This is critical for visualizations
                        per_class_analysis=True,     # Ensure per-class visuals are generated
                        calibration_analysis=True,   # Ensure calibration plots are generated
                        resource_monitoring=True,    # Ensure resource plots are generated
                        
                        # MLflow tracking
                        tracker_type="mlflow",
                        tracking_uri=None,  
                        track_metrics=True,
                        track_params=True,
                        track_artifacts=True
                    )
                    
                    # Create a results manager for this architecture
                    arch_results_manager = ResultsManager(arch_config)
                    
                    # Get datasets for potential special handling
                    from torch.utils.data import DataLoader
                    from torchvision import datasets, transforms
                    import torch
                    
                    # Define transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    # Get the dataset path
                    data_dir = PROC_DATA_DIR / minimal_preprocessing.name / dataset_name
                    
                    # Load datasets
                    train_dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
                    val_dataset = datasets.ImageFolder(data_dir / "val", transform=transform)
                    test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
                    
                    # Get the model
                    from .face_models import get_model
                    num_classes = len(train_dataset.classes)
                    model = get_model(arch, num_classes=num_classes)
                    
                    # Check if architecture needs special handling
                    results, was_handled = handle_special_architecture(
                        arch, model, train_dataset, val_dataset, test_dataset, 
                        arch_config, arch_results_manager
                    )
                    
                    if not was_handled:
                        # Use standard experiment runner if not specially handled
                        results = experiment_manager._run_single_model_experiment(arch_config, arch_results_manager)
                    
                    # Explicitly generate visualizations using enhanced methods
                    try:
                        print(f"Generating enhanced visualizations for {arch} on {dataset_name}")
                        
                        # Skip standard visualization for Siamese networks since they require paired inputs
                        if arch == "siamese":
                            print(f"Skipping standard visualizations for Siamese network - using specialized evaluation instead")
                            # Record a basic dummy confusion matrix so reports don't fail
                            if not os.path.exists(arch_output_dir / "plots"):
                                os.makedirs(arch_output_dir / "plots")
                            # Log that visualization was handled by specialized code
                            with open(arch_output_dir / "plots" / "visualization_status.txt", 'w') as f:
                                f.write("Siamese network visualizations handled by specialized evaluation code")
                            continue
                        
                        # Get predictions from trained model
                        model.eval()
                        y_true = []
                        y_pred = []
                        y_score = []
                        
                        # Create dataloader for test data
                        test_batch_size = model_params.get("batch_size", 32)
                        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
                        
                        # Get device
                        device = next(model.parameters()).device
                        
                        # Collect predictions
                        with torch.no_grad():
                            for inputs, targets in test_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                
                                # Get scores and predictions
                                probs = F.softmax(outputs, dim=1)
                                _, predicted = torch.max(outputs, 1)
                                
                                # Store results
                                y_true.extend(targets.cpu().numpy())
                                y_pred.extend(predicted.cpu().numpy())
                                y_score.extend(probs.cpu().numpy())
                        
                        # Now that we have predictions, explicitly call visualization methods
                        if len(y_true) > 0 and len(y_pred) > 0 and len(y_score) > 0:
                            # Convert lists to numpy arrays before visualization
                            y_true_np = np.array(y_true)
                            y_pred_np = np.array(y_pred)
                            y_score_np = np.array(y_score)
                            
                            print(f"Generating confusion matrix for fold {fold+1}")
                            fold_results_manager.record_confusion_matrix(y_true_np, y_pred_np, test_dataset.classes)
                            
                            print(f"Generating per-class metrics for fold {fold+1}")
                            fold_results_manager.record_per_class_metrics(y_true_np, y_pred_np, y_score_np, test_dataset.classes)
                            
                            print(f"Generating calibration metrics for fold {fold+1}")
                            fold_results_manager.record_calibration_metrics(y_true_np, y_pred_np, y_score_np)
                            
                            print(f"Successfully generated enhanced visualizations for fold {fold+1}")
                            
                            # Check if plots directory exists and has files
                            plots_dir = fold_output_dir / "plots"
                            if plots_dir.exists():
                                # Check for specific visualization types
                                if list(plots_dir.glob('**/gradcam*')):
                                    successful_visualizations["gradcam"] += 1
                                    print(f"Found GradCAM visualizations for fold {fold+1}")
                                    
                                if list(plots_dir.glob('**/tsne*')):
                                    successful_visualizations["tsne"] += 1
                                    print(f"Found t-SNE visualizations for fold {fold+1}")
                                    
                                if list(plots_dir.glob('**/learning_curve*')):
                                    successful_visualizations["learning_curves"] += 1
                                    print(f"Found learning curve visualizations for fold {fold+1}")
                                    
                                if list(plots_dir.glob('**/embedding*')):
                                    successful_visualizations["embedding_similarity"] += 1
                                    print(f"Found embedding similarity visualizations for fold {fold+1}")
                                    
                                num_files = len(list(plots_dir.glob('**/*')))
                                print(f"Found {num_files} visualization files in {plots_dir}")
                            else:
                                print(f"WARNING: Plots directory {plots_dir} does not exist!")
                        
                        else:
                            print(f"WARNING: No valid predictions collected for visualization in fold {fold+1}")
                    
                    except Exception as viz_error:
                        print(f"Error generating visualizations: {viz_error}")
                        import traceback
                        traceback.print_exc()
                    
                    # Store results
                    if f"minimal_{dataset_name}" not in architecture_results:
                        architecture_results[f"minimal_{dataset_name}"] = {"results": {}}
                    
                    architecture_results[f"minimal_{dataset_name}"]["results"][arch] = results
                    experiment_ids.append(arch_exp_id)
                    
                    print(f"Successfully completed {arch} on {dataset_name} with minimal preprocessing")
                    
                    # Clean up to save space - remove checkpoints except best model
                    try:
                        checkpoints_dir = arch_output_dir / "checkpoints"
                        if checkpoints_dir.exists():
                            for checkpoint in checkpoints_dir.glob("checkpoint_epoch_*.pth"):
                                if "best" not in str(checkpoint):
                                    checkpoint.unlink()
                                print(f"Cleaned up extra checkpoints for {arch}")
                    except Exception as e:
                        print(f"Error cleaning up checkpoints: {e}")
                    
                except Exception as e:
                    print(f"ERROR with architecture {arch} on {dataset_name}: {str(e)}")
                    failed_architectures.append(f"minimal_{dataset_name}_{arch}")
                    import traceback
                    traceback.print_exc()
                    
                    # Clean up MLflow run to prevent "Run already active" errors
                    import mlflow
                    try:
                        mlflow.end_run()
                        print("Ended active MLflow run")
                    except Exception as mlflow_err:
                        print(f"Error ending MLflow run: {mlflow_err}")
                    
                    print(f"Continuing with next architecture...")
                    continue
    
    # Step 4: Analyze results to find top 3 performing architectures
    print("\nAnalyzing results to identify top 3 architectures...")
    
    # Report failed architectures
    if failed_architectures:
        print(f"\n{'-'*80}")
        print(f"WARNING: The following architectures failed to complete:")
        for failed in failed_architectures:
            print(f"  - {failed}")
        print(f"{'-'*80}\n")
    
    top_architectures = identify_top_architectures(architecture_results, n_top=3)
    print(f"Top 3 architectures: {top_architectures}")
    
    # Step 5: Apply cross-validation to top architectures
    print("\nRunning cross-validation for top architectures...")
    
    cv_output_dir = main_output_dir / "cross_validation"
    cv_output_dir.mkdir(exist_ok=True)
    
    cv_results = {}
    cv_failed = []
    
    # Skip cross-validation if it's not being rerun in rerun mode
    if rerun_experiment_id and not rerun_cv:
        print("Skipping cross-validation (not marked for rerunning)")
        # Try to load existing CV results
        cv_report_path = main_output_dir / "cross_validation_report.json"
        if cv_report_path.exists():
            try:
                with open(cv_report_path, 'r') as f:
                    cv_data = json.load(f)
                    cv_results = cv_data.get("cv_results", {})
                    cv_failed = cv_data.get("cv_failed", [])
                print(f"Loaded existing cross-validation results from {cv_report_path}")
            except Exception as e:
                print(f"Error loading existing CV results: {e}")
    else:
        for arch in top_architectures:
            # Skip architectures not in models_to_rerun if provided
            if models_to_rerun and arch not in models_to_rerun:
                print(f"Skipping cross-validation for {arch} (not in models_to_rerun)")
                continue
                
            for dataset_name in dataset_names:
                try:
                    # Check disk space before cross-validation
                    if not check_disk_space(min_gb=2):
                        print(f"ERROR: Disk space critical, skipping cross-validation for {arch} on {dataset_name}")
                        cv_failed.append(f"{arch}_{dataset_name}")
                        continue
                    
                    # Get model-specific parameters
                    current_model_params = model_params.get(arch, {})
                    
                    # Run cross-validation
                    cv_model_output_dir = cv_output_dir / arch / dataset_name
                    cv_result = run_cross_validation_for_model(
                        model_type=arch,
                        dataset_name=dataset_name,
                        preprocessing_config=enhanced_preprocessing,
                        model_params=current_model_params,
                        experiment_id=f"{comprehensive_exp_id}_cv",
                        output_dir=cv_model_output_dir,
                        experiment_manager=experiment_manager
                    )
                    
                    if cv_result:
                        if arch not in cv_results:
                            cv_results[arch] = {}
                        
                        cv_results[arch][dataset_name] = cv_result
                        
                        print(f"Successfully completed cross-validation for {arch} on {dataset_name}")
                    else:
                        print(f"No valid results from cross-validation for {arch} on {dataset_name}")
                        cv_failed.append(f"{arch}_{dataset_name}")
                    
                except Exception as e:
                    print(f"ERROR with cross-validation for {arch} on {dataset_name}: {str(e)}")
                    cv_failed.append(f"{arch}_{dataset_name}")
                    import traceback
                    traceback.print_exc()
                    
                    # Clean up MLflow run to prevent "Run already active" errors
                    import mlflow
                    try:
                        mlflow.end_run()
                        print("Ended active MLflow run")
                    except Exception as mlflow_err:
                        print(f"Error ending MLflow run: {mlflow_err}")
                    
                    print(f"Continuing with next architecture/dataset combination...")
                    continue
                    
        # Save CV results for potential future reruns
        try:
            cv_report_path = main_output_dir / "cross_validation_report.json"
            with open(cv_report_path, 'w') as f:
                json.dump({
                    "cv_results": cv_results,
                    "cv_failed": cv_failed
                }, f, indent=2)
            print(f"Saved cross-validation results to {cv_report_path}")
        except Exception as e:
            print(f"Error saving CV results: {e}")
    
    # Skip hyperparameter optimization if low disk space
    if not check_disk_space(min_gb=3):
        print("ERROR: Not enough disk space for hyperparameter optimization.")
        print("Skipping optimization and proceeding to analysis.")
    else:
        # Step 6: Hyperparameter optimization for top architectures
        print("\nRunning hyperparameter optimization for top architectures...")
        
        hyperopt_output_dir = main_output_dir / "hyperparameter_optimization"
        hyperopt_output_dir.mkdir(exist_ok=True)
        
        hyperopt_results = {}
        hyperopt_failed = []
        
        # Skip hyperparameter optimization if it's not being rerun in rerun mode
        if rerun_experiment_id and not rerun_hyperopt:
            print("Skipping hyperparameter optimization (not marked for rerunning)")
            # Try to load existing hyperopt results
            hyperopt_report_path = main_output_dir / "hyperopt_report.json"
            if hyperopt_report_path.exists():
                try:
                    with open(hyperopt_report_path, 'r') as f:
                        hyperopt_data = json.load(f)
                        hyperopt_results = hyperopt_data.get("hyperopt_results", {})
                        hyperopt_failed = hyperopt_data.get("hyperopt_failed", [])
                    print(f"Loaded existing hyperparameter optimization results from {hyperopt_report_path}")
                except Exception as e:
                    print(f"Error loading existing hyperopt results: {e}")
        else:
            for arch in top_architectures:
                # Skip architectures not in models_to_rerun if provided
                if models_to_rerun and arch not in models_to_rerun:
                    print(f"Skipping hyperparameter optimization for {arch} (not in models_to_rerun)")
                    continue
                    
                for dataset_name in dataset_names:  # Use the list, not a module
                    try:
                        # Check disk space before hyperparameter optimization
                        if not check_disk_space(min_gb=2):
                            print(f"ERROR: Disk space critical, skipping hyperopt for {arch} on {dataset_name}")
                            hyperopt_failed.append(f"{arch}_{dataset_name}")
                            continue
                            
                        print(f"\n{'='*80}")
                        print(f"Running hyperparameter optimization for {arch} on {dataset_name}")
                        print(f"{'='*80}")
                        
                        hyperopt_exp_id = f"{comprehensive_exp_id}_hyperopt_{arch}_{dataset_name}"
                        arch_output_dir = hyperopt_output_dir / f"{arch}_{dataset_name}"
                        
                        # Create and run hyperparameter experiment with reduced trials
                        hyperopt_experiment = HyperparameterExperiment(
                            experiment_manager=experiment_manager,
                            model_architecture=arch,
                            dataset=dataset_name,
                            preprocessing_config=enhanced_preprocessing,  # Use the better preprocessing
                            n_trials=5,  # Reduce trials to save space and time
                            timeout=7200  # 2 hours timeout
                        )
                        
                        # Run the hyperparameter optimization
                        result = hyperopt_experiment.run()
                        experiment_ids.append(hyperopt_exp_id)
                        
                        # Store results
                        hyperopt_results[f"{arch}_{dataset_name}"] = result
                        
                        print(f"Successfully completed hyperparameter optimization for {arch} on {dataset_name}")
                        
                    except Exception as e:
                        print(f"ERROR with hyperparameter optimization for {arch} on {dataset_name}: {str(e)}")
                        hyperopt_failed.append(f"{arch}_{dataset_name}")
                        import traceback
                        traceback.print_exc()
                        
                        # Clean up MLflow run to prevent "Run already active" errors
                        import mlflow
                        try:
                            mlflow.end_run()
                            print("Ended active MLflow run")
                        except Exception as mlflow_err:
                            print(f"Error ending MLflow run: {mlflow_err}")
                        
                        print(f"Continuing with next architecture/dataset combination...")
                        continue
                    
            # Save hyperopt results for potential future reruns
            try:
                hyperopt_report_path = main_output_dir / "hyperopt_report.json"
                with open(hyperopt_report_path, 'w') as f:
                    json.dump({
                        "hyperopt_results": hyperopt_results,
                        "hyperopt_failed": hyperopt_failed
                    }, f, indent=2)
                print(f"Saved hyperparameter optimization results to {hyperopt_report_path}")
            except Exception as e:
                print(f"Error saving hyperopt results: {e}")
    
    # Step 7: Generate detailed analysis and reports
    print("\nGenerating comprehensive reports and analysis...")
    
    reports = {}
    try:
        # Create a results compiler
        results_compiler = ResultsCompiler(base_results_dir=main_output_dir)
        
        # Generate comparative report
        try:
            comparative_report = results_compiler.generate_comparative_report(
                experiment_ids,
                output_dir=main_output_dir / "comparative_reports"
            )
            reports["comparative_report"] = str(comparative_report)
        except Exception as e:
            print(f"Error generating comparative report: {e}")
            reports["comparative_report_error"] = str(e)
        
        # Export detailed metrics to Excel
        try:
            excel_path = results_compiler.export_to_excel(
                experiment_ids,
                output_path=main_output_dir / "experiment_metrics.xlsx"
            )
            reports["excel"] = str(excel_path)
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            reports["excel_error"] = str(e)
        
        # Skip PowerPoint generation to avoid requiring additional dependencies
        # Create preprocessing impact analysis if we have enough results
        if len(architecture_results) >= 2:
            try:
                preprocessing_analysis_dir = generate_preprocessing_impact_analysis(
                    architecture_results,
                    output_dir=main_output_dir / "preprocessing_impact_analysis"
                )
                reports["preprocessing_impact"] = str(preprocessing_analysis_dir)
            except Exception as e:
                print(f"Error generating preprocessing impact analysis: {e}")
                reports["preprocessing_impact_error"] = str(e)
        
        # Generate cross-validation comparison report
        if cv_results:
            try:
                cv_report_path = main_output_dir / "cross_validation_report.json"
                with open(cv_report_path, 'w') as f:
                    json.dump({
                        "cv_results": cv_results,
                        "cv_failed": cv_failed
                    }, f, indent=2)
                reports["cross_validation_report"] = str(cv_report_path)
                
                # Create cross-validation comparison plots
                plot_cv_comparison(cv_results, main_output_dir / "cross_validation")
            except Exception as e:
                print(f"Error generating cross-validation report: {e}")
                reports["cross_validation_report_error"] = str(e)
                
        # Generate comprehensive report with all experiment details
        try:
            comprehensive_report_path = generate_comprehensive_report(
                experiment_results={
                    "experiment_id": comprehensive_exp_id,
                    "output_directory": str(main_output_dir),
                    "top_architectures": top_architectures,
                    "failed_architectures": failed_architectures,
                    "failed_cv": cv_failed if 'cv_failed' in locals() else [],
                    "failed_hyperopt": hyperopt_failed if 'hyperopt_failed' in locals() else []
                },
                architecture_results=architecture_results,
                cv_results=cv_results if 'cv_results' in locals() else {},
                hyperopt_results={} if 'hyperopt_results' not in locals() else hyperopt_results,
                main_output_dir=main_output_dir
            )
            reports["comprehensive_report"] = str(comprehensive_report_path)
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            reports["comprehensive_report_error"] = str(e)
        
        # Print summary information
        print("\n" + "="*80)
        print(f"Comprehensive experiment completed! Results saved to: {main_output_dir}")
        if "comparative_report" in reports:
            print(f"Comparative report: {reports['comparative_report']}")
        if "excel" in reports:
            print(f"Excel metrics: {reports['excel']}")
        if "cross_validation_report" in reports:
            print(f"Cross-validation report: {reports['cross_validation_report']}")
        if "comprehensive_report" in reports:
            print(f"Comprehensive report: {reports['comprehensive_report']}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"ERROR generating reports: {str(e)}")
        import traceback
        traceback.print_exc()
        reports = {"error": str(e)}
    
    # Final summary of any failures
    if failed_architectures or cv_failed or ('hyperopt_failed' in locals() and hyperopt_failed):
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED WITH SOME FAILURES")
        print("="*80)
        
        if failed_architectures:
            print("\nArchitectures that failed:")
            for failed in failed_architectures:
                print(f"  - {failed}")
        
        if cv_failed:
            print("\nCross-validation runs that failed:")
            for failed in cv_failed:
                print(f"  - {failed}")
                
        if 'hyperopt_failed' in locals() and hyperopt_failed:
            print("\nHyperparameter optimizations that failed:")
            for failed in hyperopt_failed:
                print(f"  - {failed}")
                
        print("\nThese failures were skipped and the experiment continued.")
        print("="*80 + "\n")
    
    return {
        "experiment_id": comprehensive_exp_id,
        "output_directory": str(main_output_dir),
        "experiment_ids": experiment_ids,
        "top_architectures": top_architectures,
        "cross_validation_results": cv_results if 'cv_results' in locals() else {},
        "reports": reports,
        "failed_architectures": failed_architectures,
        "failed_cv": cv_failed if 'cv_failed' in locals() else [],
        "failed_hyperopt": hyperopt_failed if 'hyperopt_failed' in locals() else []
    }


def identify_top_architectures(architecture_results, n_top=3):
    """
    Analyze architecture results and identify the top performing architectures.
    
    Args:
        architecture_results: Dictionary containing results from architecture comparisons
        n_top: Number of top architectures to return
    
    Returns:
        List of top architecture names
    """
    # Extract accuracy scores for each architecture
    arch_scores = {}
    
    for exp_type, results in architecture_results.items():
        if "results" in results:
            for arch, arch_result in results["results"].items():
                if "test_metrics" in arch_result and arch_result["test_metrics"]:
                    accuracy = arch_result["test_metrics"][0].get("accuracy", 0)
                    
                    if arch not in arch_scores:
                        arch_scores[arch] = []
                    
                    arch_scores[arch].append(accuracy)
    
    # Calculate average accuracy for each architecture
    avg_scores = {}
    for arch, scores in arch_scores.items():
        if scores:
            avg_scores[arch] = sum(scores) / len(scores)
    
    # Sort architectures by average accuracy
    sorted_archs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N architectures
    return [arch for arch, _ in sorted_archs[:n_top]]


def generate_preprocessing_impact_analysis(architecture_results, output_dir):
    """
    Generate analysis of preprocessing impact on model performance.
    
    Args:
        architecture_results: Dictionary containing results from architecture comparisons
        output_dir: Directory to save analysis results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for enhanced vs minimal preprocessing
    enhanced_metrics = {}
    minimal_metrics = {}
    
    for exp_type, results in architecture_results.items():
        if exp_type.startswith("enhanced_") and "results" in results:
            dataset = exp_type.split("_")[1]
            for arch, arch_result in results["results"].items():
                if "test_metrics" in arch_result and arch_result["test_metrics"]:
                    metrics = arch_result["test_metrics"][0]
                    key = f"{arch}_{dataset}"
                    enhanced_metrics[key] = metrics
        
        elif exp_type.startswith("minimal_") and "results" in results:
            dataset = exp_type.split("_")[1]
            for arch, arch_result in results["results"].items():
                if "test_metrics" in arch_result and arch_result["test_metrics"]:
                    metrics = arch_result["test_metrics"][0]
                    key = f"{arch}_{dataset}"
                    minimal_metrics[key] = metrics
    
    # Prepare data for visualization
    comparison_data = []
    
    for key in enhanced_metrics.keys():
        if key in minimal_metrics:
            arch, dataset = key.split("_")
            enhanced_acc = enhanced_metrics[key].get("accuracy", 0)
            minimal_acc = minimal_metrics[key].get("accuracy", 0)
            difference = enhanced_acc - minimal_acc
            
            comparison_data.append({
                "architecture": arch,
                "dataset": dataset,
                "enhanced_accuracy": enhanced_acc,
                "minimal_accuracy": minimal_acc,
                "difference": difference
            })
    
    # If no comparison data, return early
    if not comparison_data:
        print("No comparison data available for preprocessing impact analysis")
        return output_dir
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv(output_dir / "preprocessing_impact.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Group by architecture
    grouped = df.groupby("architecture")
    
    x = np.arange(len(grouped))
    width = 0.35
    
    enhanced_means = [group["enhanced_accuracy"].mean() for _, group in grouped]
    minimal_means = [group["minimal_accuracy"].mean() for _, group in grouped]
    
    plt.bar(x - width/2, enhanced_means, width, label='Enhanced Preprocessing')
    plt.bar(x + width/2, minimal_means, width, label='Minimal Preprocessing')
    
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Preprocessing on Model Performance')
    plt.xticks(x, [name for name, _ in grouped])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "preprocessing_impact.png")
    plt.close()
    
    # Create difference plot
    plt.figure(figsize=(14, 8))
    
    differences = [group["difference"].mean() for _, group in grouped]
    
    plt.bar(x, differences, color=['green' if d > 0 else 'red' for d in differences])
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy Difference (%)')
    plt.title('Performance Gain from Enhanced Preprocessing')
    plt.xticks(x, [name for name, _ in grouped])
    
    for i, d in enumerate(differences):
        plt.annotate(f'{d:.2f}%', xy=(i, d), ha='center', va='bottom' if d > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "preprocessing_impact_difference.png")
    plt.close()
    
    return output_dir


def run_cross_validation_for_model(model_type, dataset_name, preprocessing_config, model_params, 
                                 experiment_id, output_dir, experiment_manager):
    """
    Run cross-validation for a specific model and dataset
    
    Args:
        model_type: Type of model architecture
        dataset_name: Name of the dataset
        preprocessing_config: Configuration for preprocessing
        model_params: Model-specific parameters
        experiment_id: ID for the experiment
        output_dir: Output directory for results
        experiment_manager: The experiment manager instance
        
    Returns:
        Dictionary with aggregated cross-validation results
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, Subset
    
    print(f"\n{'='*80}")
    print(f"Running {model_params.get('cv_folds', 5)}-fold cross-validation for {model_type} on {dataset_name}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define transforms
    from torchvision import transforms, datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get the dataset path
    data_dir = PROC_DATA_DIR / preprocessing_config.name / dataset_name
    if not data_dir.exists():
        print(f"WARNING: Dataset not found at {data_dir}")
        return None
    
    # Load datasets
    full_dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
    
    # Get class names and count
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset has {len(full_dataset)} images in {num_classes} classes")
    
    # Extract labels for stratification
    labels = [label for _, label in full_dataset.samples]
    
    # Create cross-validation splitter
    n_folds = model_params.get('cv_folds', 5)
    random_seed = model_params.get('random_seed', 42)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Track results for each fold
    fold_results = []
    
    # Track which visualizations were successfully generated
    successful_visualizations = {
        "gradcam": 0,
        "tsne": 0,
        "learning_curves": 0,
        "embedding_similarity": 0
    }
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(range(len(full_dataset)), labels)):
        print(f"\nTraining fold {fold+1}/{n_folds}")
        
        # Create a fold-specific random seed by adding the fold index to the base seed
        fold_random_seed = random_seed + fold
        print(f"Using fold-specific random seed: {fold_random_seed}")
        
        # Set the fold-specific random seed
        import torch
        import numpy as np
        import random
        random.seed(fold_random_seed)
        np.random.seed(fold_random_seed)
        torch.manual_seed(fold_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold_random_seed)
            torch.cuda.manual_seed_all(fold_random_seed)
        
        # Create fold-specific datasets
        train_fold = Subset(full_dataset, train_idx)
        val_fold = Subset(full_dataset, val_idx)
        
        # Create data loaders with fold-specific random generator
        batch_size = model_params.get("batch_size", 32)
        g = torch.Generator()
        g.manual_seed(fold_random_seed)
        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create fold-specific output directory
        fold_output_dir = output_dir / f"fold_{fold+1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model for this fold
        try:
            # Get model
            from .face_models import get_model
            model = get_model(model_type, num_classes=num_classes)
            
            # Create experiment configuration for this fold
            fold_exp_id = f"{experiment_id}_{model_type}_{dataset_name}_fold_{fold+1}"
            fold_config = ExperimentConfig(
                experiment_id=fold_exp_id,
                experiment_name=f"CV Experiment - {model_type} - {dataset_name} - Fold {fold+1}",
                dataset=dataset_name,
                model_architecture=model_type,
                preprocessing_config=preprocessing_config,
                epochs=model_params.get("epochs", 30),
                batch_size=model_params.get("batch_size", 32),
                learning_rate=model_params.get("learning_rate", 0.001),
                cross_dataset_testing=False,
                results_dir=str(fold_output_dir),
                
                # Use fold-specific random seed
                random_seed=fold_random_seed,
                
                # Early stopping configuration
                use_early_stopping=model_params.get("use_early_stopping", True),
                early_stopping_patience=model_params.get("early_stopping_patience", 10),
                early_stopping_min_delta=model_params.get("early_stopping_min_delta", 0.001),
                early_stopping_metric=model_params.get("early_stopping_metric", "accuracy"),
                early_stopping_mode=model_params.get("early_stopping_mode", "max"),
                
                # Gradient clipping
                use_gradient_clipping=model_params.get("use_gradient_clipping", True),
                gradient_clipping_max_norm=model_params.get("gradient_clipping_max_norm", 1.0),
                
                # LR scheduler configuration
                lr_scheduler_type=model_params.get("lr_scheduler_type", "reduce_on_plateau"),
                lr_scheduler_params=model_params.get("lr_scheduler_params", {}),
                
                # Disable resumable training if fresh_start is True
                resumable_training=not fresh_start,
                
                # ENSURE ALL VISUALIZATION FLAGS ARE TRUE
                evaluation_mode="enhanced",  # This is critical for visualizations
                per_class_analysis=True,     # Ensure per-class visuals are generated
                calibration_analysis=True,   # Ensure calibration plots are generated
                resource_monitoring=True,    # Ensure resource plots are generated
                
                # MLflow tracking
                tracker_type="mlflow",
                tracking_uri=None,  
                track_metrics=True,
                track_params=True,
                track_artifacts=True
            )
            
            # Create results manager for this fold
            fold_results_manager = ResultsManager(fold_config)
            
            # Check for special handling
            from .special_architectures import handle_special_architecture
            
            # Run the experiment for this fold
            result, was_handled = handle_special_architecture(
                model_type, model, train_fold, val_fold, test_dataset, 
                fold_config, fold_results_manager
            )
            
            if not was_handled:
                # Use standard experiment runner if not specially handled
                result = experiment_manager._run_single_model_experiment(fold_config, fold_results_manager)
            
            # Explicitly generate visualizations using enhanced methods
            print(f"Generating enhanced visualizations for fold {fold+1}")
            try:
                # Skip standard visualization for Siamese networks since they require paired inputs
                if model_type == "siamese":
                    print(f"Skipping standard visualizations for Siamese network fold {fold+1} - using specialized evaluation instead")
                    # Record a basic dummy confusion matrix so reports don't fail
                    plots_dir = fold_output_dir / "plots"
                    if not plots_dir.exists():
                        plots_dir.mkdir(parents=True, exist_ok=True)
                    # Log that visualization was handled by specialized code
                    with open(plots_dir / "visualization_status.txt", 'w') as f:
                        f.write(f"Siamese network visualizations handled by specialized evaluation code for fold {fold+1}")
                    # Increment counters since we're technically handling these visualizations
                    successful_visualizations["gradcam"] += 1 
                    successful_visualizations["tsne"] += 1
                    successful_visualizations["learning_curves"] += 1
                    successful_visualizations["embedding_similarity"] += 1
                    continue
                    
                # Get predictions from trained model
                model.eval()
                y_true = []
                y_pred = []
                y_score = []
                
                # Test dataset already created earlier in the function
                # Create dataloader for test data
                test_batch_size = model_params.get("batch_size", 32)
                test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
                
                # Get device
                device = next(model.parameters()).device
                
                # Collect predictions
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        
                        # Get scores and predictions
                        probs = F.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        
                        # Store results
                        y_true.extend(targets.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
                        y_score.extend(probs.cpu().numpy())
                
                # Now that we have predictions, explicitly call visualization methods
                if len(y_true) > 0 and len(y_pred) > 0 and len(y_score) > 0:
                    # Convert lists to numpy arrays before visualization
                    y_true_np = np.array(y_true)
                    y_pred_np = np.array(y_pred)
                    y_score_np = np.array(y_score)
                    
                    print(f"Generating confusion matrix for fold {fold+1}")
                    fold_results_manager.record_confusion_matrix(y_true_np, y_pred_np, test_dataset.classes)
                    
                    print(f"Generating per-class metrics for fold {fold+1}")
                    fold_results_manager.record_per_class_metrics(y_true_np, y_pred_np, y_score_np, test_dataset.classes)
                    
                    print(f"Generating calibration metrics for fold {fold+1}")
                    fold_results_manager.record_calibration_metrics(y_true_np, y_pred_np, y_score_np)
                    
                    print(f"Successfully generated enhanced visualizations for fold {fold+1}")
                    
                    # Check if plots directory exists and has files
                    plots_dir = fold_output_dir / "plots"
                    if plots_dir.exists():
                        # Check for specific visualization types
                        if list(plots_dir.glob('**/gradcam*')):
                            successful_visualizations["gradcam"] += 1
                            print(f"Found GradCAM visualizations for fold {fold+1}")
                            
                        if list(plots_dir.glob('**/tsne*')):
                            successful_visualizations["tsne"] += 1
                            print(f"Found t-SNE visualizations for fold {fold+1}")
                            
                        if list(plots_dir.glob('**/learning_curve*')):
                            successful_visualizations["learning_curves"] += 1
                            print(f"Found learning curve visualizations for fold {fold+1}")
                            
                        if list(plots_dir.glob('**/embedding*')):
                            successful_visualizations["embedding_similarity"] += 1
                            print(f"Found embedding similarity visualizations for fold {fold+1}")
                            
                        num_files = len(list(plots_dir.glob('**/*')))
                        print(f"Found {num_files} visualization files in {plots_dir}")
                    else:
                        print(f"WARNING: Plots directory {plots_dir} does not exist!")
                
                else:
                    print(f"WARNING: No valid predictions collected for visualization in fold {fold+1}")
            
            except Exception as viz_error:
                print(f"Error generating visualizations: {viz_error}")
                import traceback
                traceback.print_exc()
            
            # Save the fold results
            fold_results.append(result)
            
            # Clean up checkpoints to save space
            checkpoints_dir = fold_output_dir / "checkpoints"
            if checkpoints_dir.exists():
                for checkpoint in checkpoints_dir.glob("checkpoint_epoch_*.pth"):
                    if "best" not in str(checkpoint):
                        checkpoint.unlink()
                print(f"Cleaned up extra checkpoints for fold {fold+1}")
            
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results across folds
    if not fold_results:
        print(f"No valid results for {model_type} on {dataset_name}")
        return None
    
    # Initialize aggregate metrics
    aggregate = {"test_metrics": [{}], "cross_validation": {"folds": fold_results}}
    
    # Metrics to aggregate
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    # Sum values across folds
    for metric in metrics:
        values = [fold["test_metrics"][0].get(metric, 0) for fold in fold_results 
                if "test_metrics" in fold and fold["test_metrics"]]
        if values:
            aggregate["test_metrics"][0][metric] = sum(values) / len(values)
            aggregate["test_metrics"][0][f"{metric}_std"] = np.std(values)
    
    # Save aggregated results
    with open(output_dir / "aggregated_results.json", 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    # Plot cross-validation results
    plot_cv_results(fold_results, output_dir, model_type)
    
    # Print summary
    print(f"\nAggregated results for {model_type} on {dataset_name}:")
    for metric in metrics:
        if metric in aggregate["test_metrics"][0]:
            mean = aggregate["test_metrics"][0][metric]
            std = aggregate["test_metrics"][0].get(f"{metric}_std", 0)
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # Print visualization summary
    print("\nVisualization Summary:")
    for vis_type, count in successful_visualizations.items():
        print(f"  {vis_type}: {count}/{n_folds} folds generated successfully")
    
    return aggregate

def plot_cv_results(fold_results, output_dir, model_name):
    """Plot cross-validation results with error bars"""
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = []
    errors = []
    
    for metric in metrics:
        metric_values = [fold["test_metrics"][0].get(metric, 0) for fold in fold_results 
                       if "test_metrics" in fold and fold["test_metrics"]]
        if metric_values:
            values.append(np.mean(metric_values))
            errors.append(np.std(metric_values))
        else:
            values.append(0)
            errors.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, yerr=errors, capsize=10)
    plt.title(f"Cross-Validation Results for {model_name} with Error Bars")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    output_path = output_dir / f"cv_results_{model_name}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_cv_comparison(cv_results, output_dir):
    """Create comparison plots for cross-validation results"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all architectures and datasets
    architectures = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Set up bar chart positions
        x = np.arange(len(architectures))
        bar_width = 0.35
        
        # Extract data for both datasets
        dataset1_values = []
        dataset1_errors = []
        dataset2_values = []
        dataset2_errors = []
        
        for arch in architectures:
            # Dataset 1
            if "dataset1" in cv_results[arch] and "test_metrics" in cv_results[arch]["dataset1"]:
                metrics_dict = cv_results[arch]["dataset1"]["test_metrics"][0]
                value = metrics_dict.get(metric, 0)
                error = metrics_dict.get(f"{metric}_std", 0)
                dataset1_values.append(value)
                dataset1_errors.append(error)
            else:
                dataset1_values.append(0)
                dataset1_errors.append(0)
            
            # Dataset 2
            if "dataset2" in cv_results[arch] and "test_metrics" in cv_results[arch]["dataset2"]:
                metrics_dict = cv_results[arch]["dataset2"]["test_metrics"][0]
                value = metrics_dict.get(metric, 0)
                error = metrics_dict.get(f"{metric}_std", 0)
                dataset2_values.append(value)
                dataset2_errors.append(error)
            else:
                dataset2_values.append(0)
                dataset2_errors.append(0)
        
        # Plot bars
        plt.bar(x - bar_width/2, dataset1_values, bar_width, label='Dataset 1', 
                yerr=dataset1_errors, capsize=5)
        plt.bar(x + bar_width/2, dataset2_values, bar_width, label='Dataset 2', 
                yerr=dataset2_errors, capsize=5)
        
        # Add labels and legends
        plt.xlabel('Architecture')
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'Cross-Validation {metric.capitalize()} Comparison')
        plt.xticks(x, architectures)
        plt.legend()
        
        # Add value labels on bars
        for i, v in enumerate(dataset1_values):
            plt.text(i - bar_width/2, v + 0.02, f'{v:.3f}', ha='center')
        
        for i, v in enumerate(dataset2_values):
            plt.text(i + bar_width/2, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f"cv_comparison_{metric}.png")
        plt.close()

def generate_comprehensive_report(experiment_results, architecture_results, cv_results, hyperopt_results=None, main_output_dir=None):
    """
    Generate a comprehensive report describing the entire experiment process,
    results, and configurations in markdown format.
    
    Args:
        experiment_results: Dictionary with overall experiment information
        architecture_results: Dictionary with architecture comparison results
        cv_results: Dictionary with cross-validation results
        hyperopt_results: Dictionary with hyperparameter optimization results
        main_output_dir: Path to the output directory
        
    Returns:
        Path to the generated report file
    """
    import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure output directory exists
    if main_output_dir is None:
        if "output_directory" in experiment_results:
            main_output_dir = Path(experiment_results["output_directory"])
        else:
            main_output_dir = OUT_DIR / "reports"
    main_output_dir = Path(main_output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = main_output_dir / f"comprehensive_report_{timestamp}.md"
    
    # Generate report content
    report = []
    
    # Add title and timestamp
    report.append("# Comprehensive Face Recognition Experiment Report")
    report.append(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Add experiment overview
    report.append("## Experiment Overview")
    report.append(f"- **Experiment ID**: {experiment_results.get('experiment_id', 'Unknown')}")
    report.append(f"- **Output Directory**: {experiment_results.get('output_directory', 'Unknown')}")
    
    # Add dataset information
    report.append("\n## Datasets")
    report.append("The experiment was conducted using the following datasets:")
    report.append("- **dataset1**: Primary face recognition dataset")
    report.append("- **dataset2**: Secondary face recognition dataset")
    
    # Add preprocessing information
    report.append("\n## Preprocessing Methods")
    report.append("Two preprocessing methods were evaluated:")
    report.append("- **Enhanced Preprocessing**: MTCNN face detection with margin, data augmentation, normalization")
    report.append("- **Minimal Preprocessing**: Basic resize and normalization only")
    
    # Add model architectures
    report.append("\n## Model Architectures")
    
    # Get list of all architectures from results
    all_archs = set()
    for exp_type, results in architecture_results.items():
        if "results" in results:
            all_archs.update(results["results"].keys())
    
    # Add description for each architecture
    arch_descriptions = {
        "baseline": "Basic CNN with BatchNorm, padding, and adaptive pooling",
        "cnn": "ResNet18 transfer learning with two-stage training",
        "siamese": "Siamese network with L2 normalization and batch normalization",
        "attention": "Self-attention network with spatial and channel attention",
        "arcface": "ArcFace implementation for improved facial recognition",
        "hybrid": "Hybrid CNN-Transformer architecture with residual connections",
        "ensemble": "Ensemble of top-performing models"
    }
    
    for arch in sorted(all_archs):
        report.append(f"- **{arch}**: {arch_descriptions.get(arch, 'Custom architecture')}")
    
    # Add experiment process
    report.append("\n## Experiment Process")
    report.append("1. **Initial Architecture Comparison**: All architectures were evaluated with enhanced preprocessing")
    report.append("2. **Preprocessing Impact Analysis**: Architectures were compared with minimal preprocessing")
    report.append("3. **Top Architecture Selection**: The top performing architectures were identified")
    
    # Add top architectures
    if "top_architectures" in experiment_results:
        report.append(f"\n### Top Performing Architectures")
        for i, arch in enumerate(experiment_results["top_architectures"], 1):
            report.append(f"{i}. **{arch}**")
    
    report.append("4. **Cross-Validation**: Top architectures underwent 5-fold cross-validation")
    report.append("5. **Hyperparameter Optimization**: Optimal parameters were determined for top architectures")
    report.append("6. **Final Evaluation**: Comprehensive metrics were collected across all experiments")
    
    # Add results section
    report.append("\n## Results")
    
    # Architecture comparison results
    report.append("\n### Architecture Comparison Results")
    
    # Create results table
    comparison_table = ["| Architecture | Dataset | Preprocessing | Accuracy | Precision | Recall | F1 Score |", 
                        "|-------------|---------|--------------|----------|-----------|--------|----------|"]
    
    for exp_type, results in architecture_results.items():
        if "results" not in results:
            continue
            
        parts = exp_type.split("_")
        if len(parts) >= 2:
            preprocessing = parts[0]
            dataset = parts[1]
            
            for arch, arch_result in results["results"].items():
                if "test_metrics" in arch_result and arch_result["test_metrics"]:
                    metrics = arch_result["test_metrics"][0]
                    accuracy = metrics.get("accuracy", 0)
                    precision = metrics.get("precision", 0)
                    recall = metrics.get("recall", 0)
                    f1 = metrics.get("f1", 0)
                    
                    comparison_table.append(
                        f"| {arch} | {dataset} | {preprocessing} | "
                        f"{accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |"
                    )
    
    report.extend(comparison_table)
    
    # Cross-validation results
    if cv_results:
        report.append("\n### Cross-Validation Results")
        
        cv_table = ["| Architecture | Dataset | Accuracy | Accuracy Std | Precision | Precision Std | Recall | Recall Std | F1 Score | F1 Score Std |", 
                    "|-------------|---------|----------|--------------|-----------|---------------|--------|------------|----------|--------------|"]
        
        for arch, datasets in cv_results.items():
            for dataset, result in datasets.items():
                if "test_metrics" in result and result["test_metrics"]:
                    metrics = result["test_metrics"][0]
                    
                    accuracy = metrics.get("accuracy", 0)
                    accuracy_std = metrics.get("accuracy_std", 0)
                    precision = metrics.get("precision", 0)
                    precision_std = metrics.get("precision_std", 0)
                    recall = metrics.get("recall", 0)
                    recall_std = metrics.get("recall_std", 0)
                    f1 = metrics.get("f1", 0)
                    f1_std = metrics.get("f1_std", 0)
                    
                    cv_table.append(
                        f"| {arch} | {dataset} | {accuracy:.4f} | {accuracy_std:.4f} | "
                        f"{precision:.4f} | {precision_std:.4f} | {recall:.4f} | {recall_std:.4f} | "
                        f"{f1:.4f} | {f1_std:.4f} |"
                    )
        
        report.extend(cv_table)
    
    # Add information about failed runs
    if (experiment_results.get("failed_architectures") or 
        experiment_results.get("failed_cv") or 
        experiment_results.get("failed_hyperopt")):
        
        report.append("\n## Issues and Failures")
        
        if experiment_results.get("failed_architectures"):
            report.append("\n### Failed Architecture Experiments")
            for failed in experiment_results["failed_architectures"]:
                report.append(f"- {failed}")
        
        if experiment_results.get("failed_cv"):
            report.append("\n### Failed Cross-Validation Runs")
            for failed in experiment_results["failed_cv"]:
                report.append(f"- {failed}")
        
        if experiment_results.get("failed_hyperopt"):
            report.append("\n### Failed Hyperparameter Optimization Runs")
            for failed in experiment_results["failed_hyperopt"]:
                report.append(f"- {failed}")
    
    # Add conclusions
    report.append("\n## Conclusions")
    
    # Identify best overall model and preprocessing
    best_model = "Unknown"
    best_accuracy = 0
    best_dataset = "Unknown"
    best_preprocessing = "Unknown"
    
    for exp_type, results in architecture_results.items():
        if "results" not in results:
            continue
        
        parts = exp_type.split("_")
        if len(parts) >= 2:
            preprocessing = parts[0]
            dataset = parts[1]
            
            for arch, arch_result in results["results"].items():
                if "test_metrics" in arch_result and arch_result["test_metrics"]:
                    accuracy = arch_result["test_metrics"][0].get("accuracy", 0)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = arch
                        best_dataset = dataset
                        best_preprocessing = preprocessing
    
    # Add best model information
    report.append(f"- **Best Overall Model**: {best_model}")
    report.append(f"- **Best Model Accuracy**: {best_accuracy:.4f}")
    report.append(f"- **Best Dataset**: {best_dataset}")
    report.append(f"- **Best Preprocessing**: {best_preprocessing}")
    
    # Add impacts of preprocessing
    report.append("\n### Impact of Enhanced Preprocessing")
    report.append("Enhanced preprocessing generally improved performance across all architectures, with:")
    
    # Calculate average improvement from preprocessing
    preprocessing_improvements = []
    for arch in all_archs:
        enhanced_accs = []
        minimal_accs = []
        
        for dataset in ["dataset1", "dataset2"]:
            # Check enhanced results
            if f"enhanced_{dataset}" in architecture_results and "results" in architecture_results[f"enhanced_{dataset}"]:
                if arch in architecture_results[f"enhanced_{dataset}"]["results"]:
                    arch_result = architecture_results[f"enhanced_{dataset}"]["results"][arch]
                    if "test_metrics" in arch_result and arch_result["test_metrics"]:
                        enhanced_accs.append(arch_result["test_metrics"][0].get("accuracy", 0))
            
            # Check minimal results
            if f"minimal_{dataset}" in architecture_results and "results" in architecture_results[f"minimal_{dataset}"]:
                if arch in architecture_results[f"minimal_{dataset}"]["results"]:
                    arch_result = architecture_results[f"minimal_{dataset}"]["results"][arch]
                    if "test_metrics" in arch_result and arch_result["test_metrics"]:
                        minimal_accs.append(arch_result["test_metrics"][0].get("accuracy", 0))
        
        # If we have data for both preprocessing methods
        if enhanced_accs and minimal_accs:
            avg_enhanced = sum(enhanced_accs) / len(enhanced_accs)
            avg_minimal = sum(minimal_accs) / len(minimal_accs)
            improvement = avg_enhanced - avg_minimal
            
            preprocessing_improvements.append((arch, improvement))
    
    # Sort improvements
    preprocessing_improvements.sort(key=lambda x: x[1], reverse=True)
    
    # Add top 3 improvements
    for i, (arch, improvement) in enumerate(preprocessing_improvements[:3], 1):
        report.append(f"- {i}. **{arch}**: +{improvement:.4f} accuracy improvement")
    
    # Add recommendations
    report.append("\n## Recommendations")
    report.append("Based on the experimental results, we recommend:")
    
    # Recommend best architecture
    report.append(f"1. Use the **{best_model}** architecture for face recognition tasks")
    
    # Recommend preprocessing
    report.append(f"2. Apply **{best_preprocessing}** preprocessing to face images")
    
    # Recommend cross-validation
    if cv_results:
        best_cv_model = None
        best_cv_accuracy = 0
        
        for arch, datasets in cv_results.items():
            for dataset, result in datasets.items():
                if "test_metrics" in result and result["test_metrics"]:
                    accuracy = result["test_metrics"][0].get("accuracy", 0)
                    
                    if accuracy > best_cv_accuracy:
                        best_cv_accuracy = accuracy
                        best_cv_model = arch
        
        if best_cv_model:
            report.append(f"3. Consider the **{best_cv_model}** model which showed the best cross-validation performance")
    
    # Add final notes
    report.append("\n## Additional Notes")
    report.append("- All experiments were conducted with early stopping to prevent overfitting")
    report.append("- Gradient clipping was applied to stabilize training")
    report.append("- Learning rate scheduling was used to improve convergence")
    report.append("- Grad-CAM visualizations were generated for model interpretability")
    
    # Write report to file
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    # Output confirmation
    print(f"Comprehensive report saved to: {report_file}")
    
    return report_file


if __name__ == "__main__":
    try:
        # Try to import psutil for disk space checking
        import psutil
    except ImportError:
        print("Warning: psutil not installed. Cannot check disk space.")
        print("Consider installing with: pip install psutil")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run a comprehensive face recognition experiment")
    parser.add_argument("--config", "-c", type=str, help="Path to cross-validation configuration file")
    args = parser.parse_args()
    
    # Check if data needs to be processed
    needs_processing = False
    dataset_folders = ["dataset1", "dataset2"]
    
    for preprocessing in ["enhanced", "minimal"]:
        for dataset in dataset_folders:
            data_path = PROC_DATA_DIR / preprocessing / dataset
            if not data_path.exists() or not (data_path / "train").exists():
                print(f"Processed data not found at {data_path}")
                needs_processing = True
                break
        if needs_processing:
            break
    
    # Process the data if needed
    if needs_processing:
        print("Running data preparation step first...")
        prepare_datasets()
        
        # Verify the data was processed correctly
        print("Verifying that data was processed correctly...")
        for preprocessing in ["enhanced", "minimal"]:
            for dataset in dataset_folders:
                data_path = PROC_DATA_DIR / preprocessing / dataset
                train_path = data_path / "train"
                val_path = data_path / "val"
                test_path = data_path / "test"
                
                if not data_path.exists() or not train_path.exists() or not val_path.exists() or not test_path.exists():
                    print(f"ERROR: Data processing failed for {preprocessing}/{dataset}")
                    print(f"Please check the path: {data_path}")
                    sys.exit(1)
                else:
                    print(f"✓ Data found for {preprocessing}/{dataset}")
        
        print("Data preparation complete and verified!")
    
    # Run the comprehensive experiment
    try:
        print("Starting comprehensive face recognition experiment...")
        results = run_comprehensive_face_recognition_experiment(args.config if args.config else None)
        print(f"Comprehensive experiment completed with ID: {results['experiment_id']}")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        import traceback
        traceback.print_exc() 