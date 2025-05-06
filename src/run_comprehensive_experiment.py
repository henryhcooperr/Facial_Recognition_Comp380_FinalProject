#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
import datetime
import json
import torch

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


def run_comprehensive_face_recognition_experiment():
    """
    Run a comprehensive face recognition experiment that evaluates all architectures
    with different preprocessing techniques, performs hyperparameter optimization,
    and generates detailed analyses and reports.
    """
    # Check disk space before starting
    if not check_disk_space(min_gb=5):
        print("ERROR: Not enough disk space to run the experiment.")
        print("Please free up some disk space before running again.")
        return {"error": "Insufficient disk space"}
    
    # Clean up old experiment directories to free space
    cleanup_old_experiments(keep_newest=2)
    
    # Make sure the processed data exists in the expected locations
    for preprocessing in ["enhanced", "minimal"]:
        for dataset in ["dataset1", "dataset2"]:
            data_path = PROC_DATA_DIR / preprocessing / dataset
            if not data_path.exists():
                raise ValueError(f"Processed data not found at {data_path}. Please run prepare_datasets() first.")
    
    # Create timestamp for the comprehensive experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comprehensive_exp_id = f"comprehensive_experiment_{timestamp}"
    
    # Create main output directory
    main_output_dir = OUT_DIR / comprehensive_exp_id
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager()
    
    # Track experiment IDs for final reporting
    experiment_ids = []
    
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
    dataset_names = ["dataset1", "dataset2"]  # Define as list, not a module
    
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
                
                arch_config = ExperimentConfig(
                experiment_id=arch_exp_id,
                experiment_name=f"Architecture Comparison - Enhanced Preprocessing - {dataset_name} - {arch}",
                dataset=dataset_name,
                model_architecture=arch,
                preprocessing_config=enhanced_preprocessing,
                epochs=REDUCED_EPOCHS,  # Reduce epochs to save space
                batch_size=32,
                learning_rate=0.001,
                cross_dataset_testing=False,
                results_dir=str(arch_output_dir),
                evaluation_mode="standard",  # Use standard evaluation to save space
                keep_last_n_checkpoints=1,  # Only keep 1 checkpoint to save space
                per_class_analysis=True,
                calibration_analysis=True,
                resource_monitoring=True,
                # Add these lines to enable MLflow tracking
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
                    
                    arch_config = ExperimentConfig(
                        experiment_id=arch_exp_id,
                        experiment_name=f"Architecture Comparison - Minimal Preprocessing - {dataset_name} - {arch}",
                        dataset=dataset_name,
                        model_architecture=arch,
                        preprocessing_config=minimal_preprocessing,
                        epochs=REDUCED_EPOCHS,  # Reduce epochs to save space
                        batch_size=32,
                        learning_rate=0.001,
                        cross_dataset_testing=False,
                        results_dir=str(arch_output_dir),
                        evaluation_mode="standard",  # Use standard evaluation to save space
                        keep_last_n_checkpoints=1,  # Only keep 1 checkpoint to save space
                        per_class_analysis=False,   # Disable to save space
                        calibration_analysis=False, # Disable to save space
                        resource_monitoring=False   # Disable to save space
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
    
    # Skip hyperparameter optimization if low disk space
    if not check_disk_space(min_gb=3):
        print("ERROR: Not enough disk space for hyperparameter optimization.")
        print("Skipping optimization and proceeding to analysis.")
    else:
        # Step 5: Hyperparameter optimization for top architectures
        print("\nRunning hyperparameter optimization for top architectures...")
        
        hyperopt_output_dir = main_output_dir / "hyperparameter_optimization"
        hyperopt_output_dir.mkdir(exist_ok=True)
        
        hyperopt_results = {}
        hyperopt_failed = []
        
        for arch in top_architectures:
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
                    print(f"Continuing with next architecture/dataset combination...")
                    continue
    
    # Step 6: Generate detailed analysis and reports
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
        
        # Print summary information
        print("\n" + "="*80)
        print(f"Comprehensive experiment completed! Results saved to: {main_output_dir}")
        if "comparative_report" in reports:
            print(f"Comparative report: {reports['comparative_report']}")
        if "excel" in reports:
            print(f"Excel metrics: {reports['excel']}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"ERROR generating reports: {str(e)}")
        import traceback
        traceback.print_exc()
        reports = {"error": str(e)}
    
    # Final summary of any failures
    if failed_architectures or ('hyperopt_failed' in locals() and hyperopt_failed):
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED WITH SOME FAILURES")
        print("="*80)
        
        if failed_architectures:
            print("\nArchitectures that failed:")
            for failed in failed_architectures:
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
        "reports": reports,
        "failed_architectures": failed_architectures,
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


if __name__ == "__main__":
    try:
        # Try to import psutil for disk space checking
        import psutil
    except ImportError:
        print("Warning: psutil not installed. Cannot check disk space.")
        print("Consider installing with: pip install psutil")
        
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
        results = run_comprehensive_face_recognition_experiment()
        print(f"Comprehensive experiment completed with ID: {results['experiment_id']}")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        import traceback
        traceback.print_exc() 