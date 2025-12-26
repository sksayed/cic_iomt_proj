"""
Hyperparameter tuning script using Optuna with early stopping.

This script:
1. Checks if preprocessed data exists, runs preprocessing if needed
2. Loads preprocessed data
3. Runs Optuna hyperparameter optimization with pruning (early stopping)
4. Trains the best model found
5. Saves results and study information

Usage:
    python tune_hyperparameters.py --architecture dense --n_trials 50
"""
import numpy as np
import sys
import argparse
from pathlib import Path
import pickle
import subprocess

# Add project root to path (for proper package imports)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import from src package
from src.config.training_config import TrainingConfig
from src.models.hyperparameter_tuner import OptunaHyperparameterTuner
from src.models.trainer import ModelTrainer
from src.utils import get_project_paths


def check_preprocessed_data_exists():
    """
    Check if preprocessed data files exist.
    
    Returns:
    --------
    bool: True if all required files exist, False otherwise
    """
    paths = get_project_paths()
    
    required_files = [
        paths['OUTPUT_DIR'] / 'X_train.npy',
        paths['OUTPUT_DIR'] / 'X_val.npy',
        paths['OUTPUT_DIR'] / 'X_test.npy',
        paths['OUTPUT_DIR'] / 'y_train.npy',
        paths['OUTPUT_DIR'] / 'y_val.npy',
        paths['OUTPUT_DIR'] / 'y_test.npy',
        paths['MODELS_DIR'] / 'preprocessing_info.pkl'
    ]
    
    all_exist = all(f.exists() for f in required_files)
    return all_exist


def run_preprocessing_notebook():
    """
    Execute the preprocessing notebook to generate required data files.
    
    Uses nbconvert to execute the notebook programmatically.
    """
    print("=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)
    print("Preprocessed data not found. Running preprocessing notebook...")
    print("This may take several minutes depending on dataset size.")
    print("=" * 80)
    
    # Get notebook path
    base_dir = Path(__file__).parent
    notebook_path = base_dir / 'notebooks' / '02_data_preprocessing.ipynb'
    
    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Preprocessing notebook not found at: {notebook_path}\n"
            "Please ensure the notebook exists or run preprocessing manually."
        )
    
    # Execute notebook using nbconvert
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        import os
        
        print("Reading preprocessing notebook...")
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print("Executing preprocessing notebook (this may take a while)...")
        # Execute notebook
        ep = ExecutePreprocessor(timeout=7200, kernel_name='python3', allow_errors=False)
        
        # Change to base directory for proper path resolution
        original_dir = os.getcwd()
        try:
            os.chdir(base_dir)
            # Execute all cells
            ep.preprocess(nb, {'metadata': {'path': str(base_dir)}})
        except Exception as e:
            raise RuntimeError(
                f"Error executing preprocessing notebook: {str(e)}\n"
                "Please check the notebook for errors or run it manually."
            )
        finally:
            os.chdir(original_dir)
        
        print("\n[SUCCESS] Preprocessing notebook executed successfully!")
        print("=" * 80)
        
    except ImportError:
        # Fallback: use jupyter nbconvert command
        print("nbformat not available, using jupyter nbconvert command...")
        print("Executing preprocessing notebook (this may take a while)...")
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=7200',
                str(notebook_path)
            ],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(
                f"Failed to execute preprocessing notebook:\n{error_msg}\n"
                f"Please run the preprocessing notebook manually: {notebook_path}"
            )
        
        print("\n[SUCCESS] Preprocessing notebook executed successfully!")
        print("=" * 80)


def ensure_preprocessed_data():
    """
    Ensure preprocessed data exists, running preprocessing if necessary.
    """
    if not check_preprocessed_data_exists():
        print("\n[WARNING] Preprocessed data not found!")
        run_preprocessing_notebook()
        
        # Verify data was created
        if not check_preprocessed_data_exists():
            raise RuntimeError(
                "Preprocessing completed but required files are still missing.\n"
                "Please check the preprocessing notebook for errors."
            )
        print("[SUCCESS] Preprocessed data is now available!")
    else:
        print("[INFO] Preprocessed data found, skipping preprocessing step.")


def load_preprocessed_data():
    """Load preprocessed data and metadata."""
    paths = get_project_paths()
    
    print("Loading preprocessed data...")
    # Load numpy arrays
    X_train = np.load(paths['OUTPUT_DIR'] / 'X_train.npy')
    X_val = np.load(paths['OUTPUT_DIR'] / 'X_val.npy')
    X_test = np.load(paths['OUTPUT_DIR'] / 'X_test.npy')
    y_train = np.load(paths['OUTPUT_DIR'] / 'y_train.npy')
    y_val = np.load(paths['OUTPUT_DIR'] / 'y_val.npy')
    y_test = np.load(paths['OUTPUT_DIR'] / 'y_test.npy')
    
    # Load preprocessing info to get dimensions
    with open(paths['MODELS_DIR'] / 'preprocessing_info.pkl', 'rb') as f:
        preprocess_info = pickle.load(f)
    
    print(f"[INFO] Data loaded:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"  Features: {preprocess_info['n_features']}")
    print(f"  Classes: {preprocess_info['n_classes']}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), preprocess_info


def sample_data(X_train, y_train, X_val, y_val, sample_size_train=None, sample_size_val=None, random_state=42):
    """
    Sample a subset of data for faster hyperparameter tuning.
    
    Uses stratified sampling to maintain class distribution when possible.
    For large datasets, this significantly speeds up hyperparameter tuning.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    sample_size_train : int, optional
        Number of training samples to use (None = use all)
    sample_size_val : int, optional
        Number of validation samples to use (None = use all)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple: Sampled (X_train, y_train, X_val, y_val)
    """
    original_train_size = len(X_train)
    original_val_size = len(X_val)
    
    # Sample training data
    if sample_size_train is not None and sample_size_train < original_train_size:
        print(f"\n[INFO] Sampling {sample_size_train:,} training samples from {original_train_size:,}...")
        rng = np.random.RandomState(random_state)
        indices = rng.choice(original_train_size, size=sample_size_train, replace=False)
        indices = np.sort(indices)  # Sort for better memory access
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"[SUCCESS] Training set reduced to {X_train.shape[0]:,} samples ({100*X_train.shape[0]/original_train_size:.1f}%)")
    
    # Sample validation data
    if sample_size_val is not None and sample_size_val < original_val_size:
        print(f"\n[INFO] Sampling {sample_size_val:,} validation samples from {original_val_size:,}...")
        rng = np.random.RandomState(random_state)
        indices = rng.choice(original_val_size, size=sample_size_val, replace=False)
        indices = np.sort(indices)  # Sort for better memory access
        X_val = X_val[indices]
        y_val = y_val[indices]
        print(f"[SUCCESS] Validation set reduced to {X_val.shape[0]:,} samples ({100*X_val.shape[0]/original_val_size:.1f}%)")
    
    return X_train, y_train, X_val, y_val


def main():
    """Main hyperparameter tuning function."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument(
        '--architecture',
        type=str,
        default='dense',
        choices=['dense', 'cnn_1d', 'deep_dense', 'xgboost'],
        help='Model architecture to tune (xgboost supports GPU acceleration)'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Maximum time in seconds for optimization'
    )
    parser.add_argument(
        '--pruner',
        type=str,
        default='median',
        choices=['median', 'halving', 'hyperband', 'none'],
        help='Pruner type for early stopping (median=default, hyperband=often more effective)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='val_loss',
        choices=['val_loss', 'val_accuracy'],
        help='Metric to optimize (for XGBoost, val_loss maps to mlogloss)'
    )
    parser.add_argument(
        '--train_best',
        action='store_true',
        help='Train the best model after optimization'
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Name for the Optuna study (study will be persisted to SQLite DB)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (requires multiprocessing setup)'
    )
    parser.add_argument(
        '--sample_size_train',
        type=int,
        default=None,
        help='Number of training samples to use for tuning (None = use all). Recommended: 10000-50000 for large datasets'
    )
    parser.add_argument(
        '--sample_size_val',
        type=int,
        default=None,
        help='Number of validation samples to use for tuning (None = use all). Recommended: 2000-10000'
    )
    parser.add_argument(
        '--use_full_data_for_final',
        action='store_true',
        help='Use full dataset when training final best model (only if --train_best is used)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CIC IoMT 2024 - Hyperparameter Tuning with Optuna")
    print("=" * 80)
    
    # Ensure preprocessed data exists (run preprocessing if needed)
    ensure_preprocessed_data()
    
    # Load preprocessed data
    (X_train, y_train, X_val, y_val, X_test, y_test), preprocess_info = load_preprocessed_data()
    
    # Store full datasets for final training (if needed)
    X_train_full = X_train.copy()
    y_train_full = y_train.copy()
    X_val_full = X_val.copy()
    y_val_full = y_val.copy()
    
    # Sample data for hyperparameter tuning if requested
    if args.sample_size_train is not None or args.sample_size_val is not None:
        print("\n" + "=" * 80)
        print("DATA SAMPLING FOR HYPERPARAMETER TUNING")
        print("=" * 80)
        X_train, y_train, X_val, y_val = sample_data(
            X_train, y_train, X_val, y_val,
            sample_size_train=args.sample_size_train,
            sample_size_val=args.sample_size_val
        )
        print("=" * 80)
    
    # Determine optimization direction
    direction = 'minimize' if 'loss' in args.metric else 'maximize'
    
    # Initialize tuner
    print(f"\nInitializing Optuna tuner...")
    tuner = OptunaHyperparameterTuner(
        input_dim=preprocess_info['n_features'],
        n_classes=preprocess_info['n_classes'],
        architecture=args.architecture,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        pruner_type=args.pruner,
        direction=direction,
        metric=args.metric
    )
    
    # Run optimization
    print(f"\nStarting hyperparameter optimization...")
    study = tuner.optimize(
        X_train, y_train,
        X_val, y_val,
        show_progress=True
    )
    
    # Print trial summary
    summary = tuner.get_trial_summary()
    print("\n" + "=" * 80)
    print("TRIAL SUMMARY")
    print("=" * 80)
    print(f"Total trials: {summary['n_trials']}")
    print(f"Completed: {summary['n_complete']}")
    print(f"Pruned (early stopped): {summary['n_pruned']}")
    print(f"Failed: {summary['n_failed']}")
    print("=" * 80)
    
    # Save study
    tuner.save_study()
    
    # Train best model if requested
    if args.train_best:
        print("\nTraining best model with optimal hyperparameters...")
        
        # Use full dataset for final training if requested
        if args.use_full_data_for_final:
            print("\n" + "=" * 80)
            print("USING FULL DATASET FOR FINAL MODEL TRAINING")
            print("=" * 80)
            print(f"Training on full dataset: {X_train_full.shape[0]:,} samples")
            print(f"Validation on full dataset: {X_val_full.shape[0]:,} samples")
            print("=" * 80)
            best_model, history = tuner.train_best_model(
                X_train_full, y_train_full,
                X_val_full, y_val_full
            )
        else:
            # Use sampled data (same as tuning)
            print(f"\nUsing sampled data for final training:")
            print(f"  Training: {X_train.shape[0]:,} samples")
            print(f"  Validation: {X_val.shape[0]:,} samples")
            best_model, history = tuner.train_best_model(
                X_train, y_train,
                X_val, y_val
            )
        
        # Evaluate on test set
        print("\nEvaluating best model on test set...")
        if args.architecture == 'xgboost':
            # XGBoost evaluation
            from sklearn.metrics import accuracy_score, log_loss, classification_report
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_loss = log_loss(y_test, y_pred_proba)
            
            print("\n" + "=" * 80)
            print("TEST SET RESULTS (Best XGBoost Model)")
            print("=" * 80)
            print(f"Test Loss (log_loss): {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("=" * 80)
        else:
            # Keras model evaluation
            test_results = best_model.evaluate(X_test, y_test, verbose=0)
            
            print("\n" + "=" * 80)
            print("TEST SET RESULTS (Best Model)")
            print("=" * 80)
            print(f"Test Loss: {test_results[0]:.4f}")
            print(f"Test Accuracy: {test_results[1]:.4f}")
            if len(test_results) > 2:
                print(f"Test Top-K Accuracy: {test_results[2]:.4f}")
            print("=" * 80)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)
    print(f"Best {args.metric}: {tuner.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in tuner.best_trial.params.items():
        print(f"  {key}: {value}")
    print("\nTo visualize optimization results:")
    print("  import optuna")
    print(f"  study = optuna.load_study(study_name='{tuner.study.study_name}', storage=None)")
    print("  optuna.visualization.plot_optimization_history(study).show()")
    print("  optuna.visualization.plot_param_importances(study).show()")
    print("=" * 80)


if __name__ == '__main__':
    main()

