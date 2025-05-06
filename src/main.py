#!/usr/bin/env python3

import argparse
import sys
import torch
import yaml
from pathlib import Path

from .base_config import logger, CHECKPOINTS_DIR, OUT_DIR, get_tracking_config
from .data_prep import get_preprocessing_config, process_raw_data
from .training import train_model, tune_hyperparameters
from .testing import evaluate_model, predict_image
from .face_models import get_model
from .visualize import plot_tsne_embeddings, plot_attention_maps, plot_embedding_similarity
from .interactive import interactive_menu
from .experiment_manager import ExperimentConfig, ExperimentManager
from .tracking import ExperimentTracker, ExperimentDashboard

def main():
    """Main entry point for face recognition system.
    Built this to handle different commands through a unified interface.
    """
    parser = argparse.ArgumentParser(description='Face Recognition System')
    subparsers = parser.add_subparsers(dest='cmd', help='Command to run')
    
    # Interactive command 
    subparsers.add_parser('interactive', help='Run the interactive menu interface')
    
    # Preprocess command
    preproc = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preproc.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    
    # Train command
    train_p = subparsers.add_parser('train', help='Train a model')
    train_p.add_argument('--model-type', type=str, required=True, 
                            choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                            help='Type of model to train')
    train_p.add_argument('--model-name', type=str, help='Name for the trained model')
    train_p.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_p.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_p.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_p.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Tune command - added this later when I realized manual hyperparameter tuning was taking too long
    tune_p = subparsers.add_parser('tune', help='Tune hyperparameters')
    tune_p.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to tune')
    tune_p.add_argument('--n-trials', type=int, default=50, help='Number of hyperparameter trials')
    
    # Evaluate command
    eval_p = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_p.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to evaluate')
    eval_p.add_argument('--model-name', type=str, help='Name of the model to evaluate')
    
    # Predict command
    pred_p = subparsers.add_parser('predict', help='Predict on a single image')
    pred_p.add_argument('--model-type', type=str, required=True, 
                              choices=['baseline', 'cnn', 'attention', 'arcface', 'hybrid'],
                              help='Type of model to use (not siamese)')
    pred_p.add_argument('--model-name', type=str, help='Name of the model to use')
    pred_p.add_argument('--image-path', type=str, required=True, help='Path to the image to predict')
    
    # Experiment command
    experiment_p = subparsers.add_parser('experiment', help='Run an experiment using a YAML/JSON configuration file')
    experiment_p.add_argument('--config', type=str, required=True, help='Path to experiment configuration file (.yaml or .json)')
    experiment_p.add_argument('--track', choices=['mlflow', 'wandb', 'none'], help='Override tracking system')
    experiment_p.add_argument('--no-track', action='store_true', help='Disable experiment tracking')
    
    # Generate config command
    gen_config_p = subparsers.add_parser('generate-config', help='Generate a template experiment configuration file')
    gen_config_p.add_argument('--output', type=str, required=True, help='Path for the output configuration file (.yaml or .json)')
    gen_config_p.add_argument('--type', type=str, choices=['single', 'comparison', 'cross-dataset'], default='single',
                             help='Type of experiment configuration to generate')
    
    # Dashboard command - for accessing experiment tracking dashboards
    dashboard_p = subparsers.add_parser('dashboard', help='Open experiment tracking dashboard')
    dashboard_p.add_argument('--type', choices=['mlflow', 'wandb'], default='mlflow', 
                           help='Type of dashboard to open')
    dashboard_p.add_argument('--compare', type=str, nargs='+', 
                          help='List of run IDs to compare')
    dashboard_p.add_argument('--metrics', type=str, nargs='+', 
                          help='List of metrics to compare across runs')
    
    # Check GPU command
    subparsers.add_parser('check-gpu', help='Check GPU availability')
    
    # List models command
    list_models_p = subparsers.add_parser('list-models', help='List available trained models')
    list_models_p.add_argument('--runs', action='store_true', help='List tracked experiment runs')
    list_models_p.add_argument('--tracker', choices=['mlflow', 'wandb'], default='mlflow', help='Tracker to use for listing runs')
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if args.cmd is None:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.cmd == 'interactive':
        return interactive_menu()
    
    elif args.cmd == 'preprocess':
        # FIXME: sometimes face detection fails on certain images - need to add better error handling
        config = get_preprocessing_config()
        process_raw_data(config, test_mode=args.test)
    
    elif args.cmd == 'train':
        # training params I found work well after lots of experiments
        # smaller batch size tends to work better but is slower
        train_model(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    elif args.cmd == 'tune':
        # Interactive selection of dataset path
        from .training import PROC_DATA_DIR
        
        # List available processed datasets
        proc_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not proc_dirs:
            raise ValueError("No processed datasets found. Process raw data first.")
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(proc_dirs, 1):
            print(f"{i}. {d.name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for tuning: ")
            try:
                idx = int(dataset_choice) - 1
                if 0 <= idx < len(proc_dirs):
                    data_dir = proc_dirs[idx]
                    break
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # My old approach was manual grid search but Optuna is way better
        best_params = tune_hyperparameters(
            model_type=args.model_type,
            dataset_path=data_dir,
            n_trials=args.n_trials
        )
        print(f"Best hyperparameters: {best_params}")
    
    elif args.cmd == 'evaluate':
        # WARNING: evaluation on validation set can be optimistic
        # Always check test set metrics before drawing conclusions
        metrics = evaluate_model(
            model_type=args.model_type,
            model_name=args.model_name
        )
    
    elif args.cmd == 'predict':
        name, conf = predict_image(
            model_type=args.model_type,
            image_path=args.image_path,
            model_name=args.model_name
        )
        print(f"Prediction: {name} (confidence: {conf:.2f})")
    
    elif args.cmd == 'experiment':
        # Run an experiment using the configuration file
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return 1
        
        print(f"Running experiment with configuration from: {config_path}")
        
        # Load the experiment configuration
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = ExperimentConfig.load_yaml(config_path)
        else:
            config = ExperimentConfig.load(config_path)
            
        # Apply tracking configuration from base_config and command line args
        tracking_config = get_tracking_config()
        
        # Handle command line overrides
        if args.no_track:
            config.tracker_type = "none"
        elif args.track:
            config.tracker_type = args.track
        elif hasattr(config, 'tracker_type') and config.tracker_type == "none":
            # If config specifically disables tracking, respect that
            pass
        else:
            # Apply default tracking settings from base_config
            config.tracker_type = tracking_config["tracker_type"]
            config.tracking_uri = tracking_config["mlflow_tracking_uri"]
            config.wandb_project = tracking_config["wandb_project"]
            config.wandb_entity = tracking_config["wandb_entity"]
            config.track_metrics = tracking_config["track_metrics"]
            config.track_params = tracking_config["track_params"]
            config.track_artifacts = tracking_config["track_artifacts"]
            config.track_models = tracking_config["track_models"]
            config.register_best_model = tracking_config["register_best_model"]
            
        # Create experiment manager and run the experiment
        experiment_manager = ExperimentManager()
        result = experiment_manager.run_experiment(config)
        
        # Print summary
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {result.get('results_dir', 'N/A')}")
        
        # Print key metrics if available
        if "test_metrics" in result and result["test_metrics"]:
            print("\nTest Metrics:")
            metrics = result["test_metrics"][0]
            for key, value in metrics.items():
                print(f"  {key}: {value}")
                
        # If dashboard URL is available, print it
        if "dashboard_url" in result and result["dashboard_url"]:
            print(f"\nDashboard URL: {result['dashboard_url']}")
            print("Open this URL in your browser to view experiment details.")
    
    elif args.cmd == 'generate-config':
        # Generate a template configuration file
        output_path = Path(args.output)
        
        # Get tracking configuration from base_config
        tracking_config = get_tracking_config()
        
        # Create a template configuration based on the type
        if args.type == 'single':
            # Single model experiment
            config = ExperimentConfig(
                experiment_name="Single Model Experiment",
                dataset=ExperimentConfig.Dataset.BOTH,
                model_architecture=ExperimentConfig.ModelArchitecture.CNN,
                epochs=30,
                batch_size=32,
                learning_rate=0.001,
                cross_dataset_testing=False,
                tracker_type=tracking_config["tracker_type"],
                tracking_uri=tracking_config["mlflow_tracking_uri"],
                wandb_project=tracking_config["wandb_project"],
                wandb_entity=tracking_config["wandb_entity"],
                track_metrics=tracking_config["track_metrics"],
                track_params=tracking_config["track_params"],
                track_artifacts=tracking_config["track_artifacts"],
                track_models=tracking_config["track_models"],
                register_best_model=tracking_config["register_best_model"]
            )
        elif args.type == 'comparison':
            # Architecture comparison experiment
            config = ExperimentConfig(
                experiment_name="Architecture Comparison",
                dataset=ExperimentConfig.Dataset.BOTH,
                model_architecture=["baseline", "cnn", "attention", "arcface"],
                epochs=30,
                batch_size=32,
                learning_rate=0.001,
                cross_dataset_testing=False,
                tracker_type=tracking_config["tracker_type"],
                tracking_uri=tracking_config["mlflow_tracking_uri"],
                wandb_project=tracking_config["wandb_project"],
                wandb_entity=tracking_config["wandb_entity"]
            )
        elif args.type == 'cross-dataset':
            # Cross-dataset experiment
            config = ExperimentConfig(
                experiment_name="Cross-Dataset Experiment",
                dataset=ExperimentConfig.Dataset.BOTH,
                model_architecture=ExperimentConfig.ModelArchitecture.CNN,
                epochs=30,
                batch_size=32,
                learning_rate=0.001,
                cross_dataset_testing=True,
                tracker_type=tracking_config["tracker_type"],
                tracking_uri=tracking_config["mlflow_tracking_uri"],
                wandb_project=tracking_config["wandb_project"],
                wandb_entity=tracking_config["wandb_entity"]
            )
        
        # Save the configuration file in the appropriate format
        if output_path.suffix.lower() in ['.yml', '.yaml']:
            config.save_yaml(output_path)
        else:
            config.save(output_path)
            
        print(f"Configuration template saved to: {output_path}")
    
    elif args.cmd == 'dashboard':
        # Open experiment tracking dashboard or run dashboard operations
        if args.type == 'mlflow':
            from .base_config import MLFLOW_TRACKING_URI
            
            # Create tracker and initialize
            tracker = ExperimentTracker.create("mlflow", tracking_uri=MLFLOW_TRACKING_URI)
            tracker.initialize("Face Recognition", tracking_uri=MLFLOW_TRACKING_URI)
            
            # Create dashboard interface
            dashboard = ExperimentDashboard(tracker)
            
            # If comparing runs is requested
            if args.compare and args.metrics:
                # Create comparison visualization
                print(f"Comparing runs: {args.compare}")
                print(f"Metrics: {args.metrics}")
                
                # Generate output path
                output_path = OUT_DIR / "visualizations" / f"run_comparison_{','.join(args.compare)}.png"
                
                # Create comparison
                fig = dashboard.compare_metrics(args.compare, args.metrics, output_path)
                
                print(f"Comparison saved to: {output_path}")
                
                # Return dashboard URL
                url = dashboard.get_dashboard_url()
                if url:
                    print(f"MLflow dashboard URL: {url}")
                else:
                    # If no HTTP URL is available, provide instructions for local server
                    print("\nTo view MLflow dashboard locally, run:")
                    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
            else:
                # Just provide the URL if no comparison is requested
                # Get recent runs to list
                runs = dashboard.get_recent_runs(limit=5)
                if runs:
                    print("\nRecent experiment runs:")
                    for i, run in enumerate(runs, 1):
                        run_id = run.get('run_id', 'unknown')
                        run_name = run.get('tags', {}).get('mlflow.runName', run_id)
                        print(f"  {i}. {run_name} (ID: {run_id})")
                
                # Return dashboard URL
                url = dashboard.get_dashboard_url()
                if url:
                    print(f"\nMLflow dashboard URL: {url}")
                else:
                    # If no HTTP URL is available, provide instructions for local server
                    print("\nTo view MLflow dashboard locally, run:")
                    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
                    
        elif args.type == 'wandb':
            from .base_config import WANDB_PROJECT, WANDB_ENTITY
            
            # Check if wandb is available
            try:
                import wandb
            except ImportError:
                print("Weights & Biases (wandb) is not installed. Please install it with:")
                print("  pip install wandb")
                return 1
                
            # Provide W&B dashboard URL
            entity = WANDB_ENTITY or wandb.api.default_entity
            project = WANDB_PROJECT
            
            if entity:
                url = f"https://wandb.ai/{entity}/{project}"
            else:
                url = f"https://wandb.ai/username/{project}"
                print("Note: Please replace 'username' with your W&B username in the URL below:")
                
            print(f"\nWeights & Biases dashboard URL: {url}")
            
            # If comparing runs is requested, provide instructions
            if args.compare and args.metrics:
                print("\nTo compare runs in W&B, open the URL above and use the W&B interface to compare runs.")
    
    elif args.cmd == 'check-gpu':
        print("GPU availability:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    elif args.cmd == 'list-models':
        if args.runs:
            # List tracked experiment runs
            if args.tracker == 'mlflow':
                from .base_config import MLFLOW_TRACKING_URI
                
                # Create tracker and initialize
                tracker = ExperimentTracker.create("mlflow", tracking_uri=MLFLOW_TRACKING_URI)
                tracker.initialize("Face Recognition", tracking_uri=MLFLOW_TRACKING_URI)
                
                # Create dashboard interface
                dashboard = ExperimentDashboard(tracker)
                
                # Get recent runs
                runs = dashboard.get_recent_runs(limit=10)
                
                # Display runs
                if runs:
                    print("\nRecent MLflow experiment runs:")
                    for i, run in enumerate(runs, 1):
                        run_id = run.get('run_id', 'unknown')
                        run_name = run.get('tags', {}).get('mlflow.runName', run_id)
                        metrics = {k: v for k, v in run.get('metrics', {}).items() if k.startswith('test_')}
                        metrics_str = ", ".join([f"{k.replace('test_', '')}={v:.3f}" for k, v in metrics.items()])
                        print(f"  {i}. {run_name} (ID: {run_id})")
                        if metrics_str:
                            print(f"     Metrics: {metrics_str}")
                else:
                    print("No MLflow runs found.")
                    
            elif args.tracker == 'wandb':
                from .base_config import WANDB_PROJECT, WANDB_ENTITY
                
                # Check if wandb is available
                try:
                    import wandb
                    from wandb.apis.public import Api
                except ImportError:
                    print("Weights & Biases (wandb) is not installed. Please install it with:")
                    print("  pip install wandb")
                    return 1
                
                # Create API client
                api = Api()
                
                # Get entity and project
                entity = WANDB_ENTITY or wandb.api.default_entity
                project = WANDB_PROJECT
                
                # Get runs
                if entity:
                    runs = api.runs(f"{entity}/{project}", per_page=10)
                else:
                    print("No W&B entity configured. Please set WANDB_ENTITY in base_config.py")
                    return 1
                
                # Display runs
                if runs:
                    print(f"\nRecent W&B experiment runs in {entity}/{project}:")
                    for i, run in enumerate(runs, 1):
                        print(f"  {i}. {run.name} (ID: {run.id})")
                        metrics = {k: v for k, v in run.summary.items() if k.startswith('test_')}
                        if metrics:
                            metrics_str = ", ".join([f"{k.replace('test_', '')}={v:.3f}" for k, v in metrics.items()])
                            print(f"     Metrics: {metrics_str}")
                else:
                    print(f"No W&B runs found in {entity}/{project}.")
        else:
            # List trained models in checkpoints directory
            model_dirs = list(CHECKPOINTS_DIR.glob('*'))
            if not model_dirs:
                print("No trained models found.")
                return 0
            
            print("\nAvailable trained models:")
            for model_dir in sorted(model_dirs):
                model_path = model_dir / 'best_model.pth'
                if model_path.exists():
                    print(f"  {model_dir.name}")
    
    return 0

# This was the original approach before I reorganized the code
# def list_datasets():
#     from .training import PROC_DATA_DIR
#     ds_dirs = list(PROC_DATA_DIR.glob('*'))
#     if not ds_dirs:
#         print("No processed datasets found.")
#         return
#     
#     print("\nAvailable datasets:")
#     for i, ds in enumerate(ds_dirs, 1):
#         if (ds / 'train').exists():
#             print(f"  {i}. {ds.name}")

if __name__ == '__main__':
    sys.exit(main()) 