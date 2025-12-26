"""
Example script to train the CIC IoMT model using class-based training modules.

This script demonstrates how to:
1. Load preprocessed data
2. Create training configuration
3. Initialize and train the model
4. Evaluate on test set
5. Save the trained model

Usage:
    python train_model.py
"""
import numpy as np
import sys
from pathlib import Path
import pickle

# Add project root to path (for proper package imports)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import from src package
from src.config.training_config import TrainingConfig
from src.models.trainer import ModelTrainer
from src.utils import get_project_paths


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
    
    print(f"✓ Data loaded:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"  Features: {preprocess_info['n_features']}")
    print(f"  Classes: {preprocess_info['n_classes']}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), preprocess_info


def create_training_config(preprocess_info):
    """Create training configuration from preprocessing info."""
    config = TrainingConfig(
        # Model architecture (from preprocessing)
        input_dim=preprocess_info['n_features'],
        n_classes=preprocess_info['n_classes'],
        
        # Architecture hyperparameters
        hidden_layers=[256, 128, 64],  # Adjust based on your needs
        activation='relu',
        output_activation='softmax',
        dropout_rate=0.3,
        
        # Training hyperparameters
        batch_size=64,  # Adjust based on your GPU memory
        epochs=100,
        learning_rate=0.001,
        optimizer='adam',
        
        # Validation
        validation_split=0.0,  # We already have validation set
        
        # Callbacks
        early_stopping_patience=15,
        reduce_lr_patience=5,
        reduce_lr_factor=0.5,
        min_lr=1e-7,
        
        # Model saving
        save_best_only=True,
        monitor_metric='val_loss',
        
        # Logging
        verbose=1
    )
    
    return config


def main():
    """Main training function."""
    print("=" * 80)
    print("CIC IoMT 2024 - Model Training")
    print("=" * 80)
    
    # Load preprocessed data
    (X_train, y_train, X_val, y_val, X_test, y_test), preprocess_info = load_preprocessed_data()
    
    # Create training configuration
    print("\nCreating training configuration...")
    config = create_training_config(preprocess_info)
    print(f"✓ Configuration created:")
    print(f"  Architecture: Dense Network")
    print(f"  Hidden layers: {config.hidden_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ModelTrainer(config)
    
    # Build and train model
    print("\nBuilding model...")
    trainer.build_model(architecture='dense')
    
    print("\nStarting training...")
    print("-" * 80)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        architecture='dense'
    )
    print("-" * 80)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    if test_metrics['top_k_accuracy']:
        print(f"Test Top-K Accuracy: {test_metrics['top_k_accuracy']:.4f}")
    print("=" * 80)
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {config.model_save_path}")
    print(f"TensorBoard logs: {config.tensorboard_log_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {config.tensorboard_log_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

