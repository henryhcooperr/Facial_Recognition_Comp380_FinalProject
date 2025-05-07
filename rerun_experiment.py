#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
import shutil
import glob

# Import your experiment functions
from src.run_comprehensive_experiment import run_comprehensive_face_recognition_experiment
from src.base_config import OUT_DIR

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run or rerun face recognition experiments")
    parser.add_argument("--mode", choices=["new", "rerun"], default=None, 
                        help="Run new experiment or rerun an existing one")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="ID of experiment to rerun (required for rerun mode)")
    parser.add_argument("--rerun-models", nargs="+", default=[], 
                        help="Specific model architectures to rerun (e.g., siamese)")
    parser.add_argument("--rerun-cv", action="store_true", 
                        help="Force rerun of cross-validation")
    parser.add_argument("--rerun-hyperopt", action="store_true", 
                        help="Force rerun of hyperparameter optimization")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Automatically confirm directory removal without prompting")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Force fresh training by disabling checkpoint resumption")
    args = parser.parse_args()

    # Interactive mode if not specified via command line
    if args.mode is None:
        mode = input("Run [n]ew experiment or [r]erun existing one? (n/r): ").strip().lower()
        args.mode = "new" if mode.startswith("n") else "rerun"

    # Handle new experiment case
    if args.mode == "new":
        print("Starting new comprehensive experiment...")
        run_comprehensive_face_recognition_experiment()
        return

    # For rerun, we need an experiment ID
    if args.experiment_id is None:
        # List available experiments
        experiments = [d for d in OUT_DIR.iterdir() if d.is_dir() and 
                      (d.name.startswith("experiment_") or d.name.startswith("comprehensive_experiment_"))]
        
        if not experiments:
            print("No existing experiments found!")
            return
        
        print("\nAvailable experiments:")
        for i, exp in enumerate(sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True), 1):
            print(f"{i}. {exp.name}")
        
        choice = int(input("\nSelect experiment number to rerun: "))
        args.experiment_id = sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)[choice-1].name

    # Locate experiment directory
    experiment_dir = OUT_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return

    print(f"Preparing to rerun experiment: {args.experiment_id}")
    
    # Find config file
    config_path = experiment_dir / "cv_config.yaml"
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        should_continue = input("Continue without config? (y/n): ").strip().lower()
        if not should_continue.startswith("y"):
            return
        config_path = None
    else:
        print(f"Found configuration file: {config_path}")

    # Check which parts to rerun
    if not args.rerun_models:
        print("\nWhich models would you like to rerun? (comma-separated, or 'all')")
        print("Options: baseline, cnn, siamese, attention, arcface, hybrid, ensemble")
        rerun_input = input("Models to rerun: ").strip().lower()
        if rerun_input == "all":
            args.rerun_models = ["baseline", "cnn", "siamese", "attention", "arcface", "hybrid", "ensemble"]
        else:
            args.rerun_models = [m.strip() for m in rerun_input.split(",")]

    if not args.rerun_cv and not args.rerun_hyperopt:
        cv_choice = input("Rerun cross-validation? (y/n): ").strip().lower()
        args.rerun_cv = cv_choice.startswith("y")
        
        hyperopt_choice = input("Rerun hyperparameter optimization? (y/n): ").strip().lower()
        args.rerun_hyperopt = hyperopt_choice.startswith("y")
    
    if not args.fresh_start:
        fresh_start = input("Start training from scratch (don't resume from checkpoints)? (y/n): ").strip().lower()
        args.fresh_start = fresh_start.startswith("y")

    # Allow user to remove specific directories to force rerun
    print("\n" + "="*80)
    print("DIRECTORY CLEANUP FOR RERUN")
    print("="*80)
    print("To ensure clean results, we need to remove old data for components being rerun.")
    
    to_remove = []
    
    # Check model directories - use glob to find all matching directories
    for model in args.rerun_models:
        # Recursively find all directories containing the model name
        model_paths = glob.glob(f"{experiment_dir}/**/*/siamese", recursive=True)
        for model_path in model_paths:
            path_obj = Path(model_path)
            rel_path = path_obj.relative_to(experiment_dir)
            to_remove.append((path_obj, str(rel_path)))
            
            # Don't add checkpoints directory separately since we're deleting the parent
            # checkpoints_dir = path_obj / "checkpoints"
            # if checkpoints_dir.exists():
            #    to_remove.append((checkpoints_dir, f"{rel_path}/checkpoints"))
                
        # Also search for any .json files containing the model name
        result_files = glob.glob(f"{experiment_dir}/**/*{model}*.json", recursive=True)
        for result_file in result_files:
            file_obj = Path(result_file)
            # Skip if the file is inside a directory we're already deleting
            if any(str(file_obj).startswith(str(dir_path)) for dir_path, _ in to_remove if dir_path.is_dir()):
                continue
            rel_path = file_obj.relative_to(experiment_dir)
            to_remove.append((file_obj, str(rel_path)))

    # Check cross-validation directory
    if args.rerun_cv:
        cv_dir = experiment_dir / "cross_validation"
        if cv_dir.exists():
            if args.rerun_models:
                # Only remove specific model CV dirs
                for model in args.rerun_models:
                    # Use glob to find all CV directories for this model
                    cv_model_paths = glob.glob(f"{cv_dir}/**/{model}", recursive=True)
                    for cv_model_path in cv_model_paths:
                        path_obj = Path(cv_model_path)
                        # Skip if this directory is already inside a directory we're deleting
                        if any(str(path_obj).startswith(str(dir_path)) for dir_path, _ in to_remove if dir_path.is_dir()):
                            continue
                        rel_path = path_obj.relative_to(experiment_dir)
                        to_remove.append((path_obj, str(rel_path)))
                    
                    # CV results files
                    cv_results_files = glob.glob(f"{cv_dir}/**/*{model}*.json", recursive=True)
                    for cv_result_file in cv_results_files:
                        file_obj = Path(cv_result_file)
                        # Skip if the file is inside a directory we're already deleting
                        if any(str(file_obj).startswith(str(dir_path)) for dir_path, _ in to_remove if dir_path.is_dir()):
                            continue
                        rel_path = file_obj.relative_to(experiment_dir)
                        to_remove.append((file_obj, str(rel_path)))
            else:
                # Remove entire CV dir
                to_remove.append((cv_dir, "cross_validation"))
        
        # Also remove cross-validation report if rerunning CV
        cv_report_path = experiment_dir / "cross_validation_report.json"
        if cv_report_path.exists():
            to_remove.append((cv_report_path, "cross_validation_report.json"))

    # Check hyperopt directory
    if args.rerun_hyperopt:
        hyperopt_dir = experiment_dir / "hyperparameter_optimization"
        if hyperopt_dir.exists():
            if args.rerun_models:
                # Only remove specific model hyperopt dirs
                for model in args.rerun_models:
                    # Use glob to find all hyperopt directories for this model
                    hyperopt_model_paths = glob.glob(f"{hyperopt_dir}/**/*{model}*", recursive=True)
                    for hyperopt_model_path in hyperopt_model_paths:
                        path_obj = Path(hyperopt_model_path)
                        # Skip if this directory is already inside a directory we're deleting
                        if any(str(path_obj).startswith(str(dir_path)) for dir_path, _ in to_remove if dir_path.is_dir()):
                            continue
                        rel_path = path_obj.relative_to(experiment_dir)
                        to_remove.append((path_obj, str(rel_path)))
            else:
                # Remove entire hyperopt dir
                to_remove.append((hyperopt_dir, "hyperparameter_optimization"))
    
    # Display directories that will be removed
    if to_remove:
        print("\nThe following directories/files will be removed for rerunning:")
        for _, path_str in to_remove:
            print(f"  - {path_str}")
        
        # Auto-confirm or ask for confirmation
        if args.auto_confirm:
            confirm = "y"
            print("Auto-confirming directory removal (--auto-confirm flag set)")
        else:
            confirm = input("\n⚠️  WARNING: This will permanently delete data! Confirm deletion? (y/n): ").strip().lower()
        
        if confirm.startswith("y"):
            print("\nRemoving old data...")
            for dir_path, _ in to_remove:
                print(f"Removing {dir_path}...")
                try:
                    if dir_path.exists():
                        if dir_path.is_file():
                            dir_path.unlink()  # Delete file
                        else:
                            shutil.rmtree(dir_path)  # Delete directory
                    else:
                        print(f"  - Skipping {dir_path} (already removed)")
                except Exception as e:
                    print(f"  - Error removing {dir_path}: {e}")
            print("✅ Old data successfully removed.")
        else:
            print("❌ Deletion cancelled.")
            should_continue = input("Continue with rerun anyway? This may affect results. (y/n): ").strip().lower()
            if not should_continue.startswith("y"):
                return
    else:
        print("No directories found matching the selected components to rerun.")

    # Run experiment with skip_completed=True
    print("\nStarting experiment rerun...")
    run_modified_experiment(
        config_path=config_path,
        experiment_id=args.experiment_id,
        models_to_rerun=args.rerun_models,
        rerun_cv=args.rerun_cv,
        rerun_hyperopt=args.rerun_hyperopt,
        fresh_start=args.fresh_start
    )

def run_modified_experiment(config_path=None, experiment_id=None, 
                           models_to_rerun=None, rerun_cv=False, rerun_hyperopt=False, fresh_start=False):
    """
    Run comprehensive experiment with ability to skip completed parts
    """
    from src.run_comprehensive_experiment import run_comprehensive_face_recognition_experiment
    
    # This will run the experiment, skipping parts that are already completed
    # and weren't explicitly marked for rerunning
    results = run_comprehensive_face_recognition_experiment(
        config_path=config_path,
        rerun_experiment_id=experiment_id,
        models_to_rerun=models_to_rerun,
        rerun_cv=rerun_cv,
        rerun_hyperopt=rerun_hyperopt,
        fresh_start=fresh_start
    )
    
    print("\nExperiment rerun completed!")
    return results

if __name__ == "__main__":
    main()
