"""
Model training class for CIC IoMT dataset.
Handles training, evaluation, saving, and loading of models.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import pickle
from typing import Tuple, Optional, Dict, Any
import joblib

from ..config.training_config import TrainingConfig
from .architectures import ModelBuilder
from ..utils import get_project_paths


class ModelTrainer:
    """Main class for training and managing deep learning models."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Parameters:
        -----------
        config : TrainingConfig
            Training configuration object
        """
        self.config = config
        self.paths = get_project_paths()
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict] = None
        
        # Setup paths
        if self.config.model_save_path is None:
            self.config.model_save_path = self.paths['MODELS_DIR'] / 'best_model.h5'
        
        if self.config.tensorboard_log_dir is None:
            self.config.tensorboard_log_dir = self.paths['RESULTS_DIR'] / 'logs'
            self.config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self, architecture: str = 'dense') -> keras.Model:
        """
        Build model based on configuration.
        
        Parameters:
        -----------
        architecture : str
            Architecture type ('dense', 'cnn_1d', or 'deep_dense')
        
        Returns:
        --------
        keras.Model: Compiled model
        """
        # Select optimizer
        if self.config.optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=self.config.learning_rate, momentum=0.9)
        elif self.config.optimizer.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Build model
        if architecture == 'dense':
            self.model = ModelBuilder.build_dense_network(
                input_dim=self.config.input_dim,
                n_classes=self.config.n_classes,
                hidden_layers=self.config.hidden_layers,
                activation=self.config.activation,
                output_activation=self.config.output_activation,
                dropout_rate=self.config.dropout_rate
            )
        elif architecture == 'cnn_1d':
            self.model = ModelBuilder.build_cnn_1d(
                input_dim=self.config.input_dim,
                n_classes=self.config.n_classes,
                activation=self.config.activation,
                output_activation=self.config.output_activation,
                dropout_rate=self.config.dropout_rate
            )
        elif architecture == 'deep_dense':
            self.model = ModelBuilder.build_deep_dense(
                input_dim=self.config.input_dim,
                n_classes=self.config.n_classes,
                hidden_layers=self.config.hidden_layers,
                activation=self.config.activation,
                output_activation=self.config.output_activation,
                dropout_rate=self.config.dropout_rate
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose from: 'dense', 'cnn_1d', 'deep_dense'")
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return self.model
    
    def _create_callbacks(self) -> list:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpoint
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.model_save_path),
                monitor=self.config.monitor_metric,
                save_best_only=self.config.save_best_only,
                verbose=1,
                mode='min' if 'loss' in self.config.monitor_metric else 'max'
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=self.config.monitor_metric,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.monitor_metric,
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            )
        )
        
        # TensorBoard
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(self.config.tensorboard_log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        architecture: str = 'dense'
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features (if None, uses validation_split)
        y_val : np.ndarray, optional
            Validation labels (if None, uses validation_split)
        architecture : str
            Model architecture type
        
        Returns:
        --------
        keras.callbacks.History: Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(architecture=architecture)
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Print model summary
        if self.config.verbose >= 1:
            print("\n" + "="*80)
            print("MODEL ARCHITECTURE")
            print("="*80)
            self.model.summary()
            print("="*80 + "\n")
        
        # Train model
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = self.config.validation_split
        
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=self.config.verbose
        )
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load_model().")
        
        results = self.model.evaluate(X_test, y_test, verbose=self.config.verbose)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_k_accuracy': results[2] if len(results) > 2 else None
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        
        Returns:
        --------
        np.ndarray: Predictions (probabilities or class indices)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load_model().")
        
        return self.model.predict(X, verbose=self.config.verbose)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        
        Returns:
        --------
        np.ndarray: Predicted class indices
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, filepath: Optional[Path] = None):
        """
        Save model and training configuration.
        
        Parameters:
        -----------
        filepath : Path, optional
            Path to save the model. If None, uses config.model_save_path
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if filepath is None:
            filepath = self.config.model_save_path
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(filepath))
        
        # Save config
        config_path = filepath.parent / f"{filepath.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training history if available
        if self.history is not None:
            history_path = filepath.parent / f"{filepath.stem}_history.json"
            history_dict = {key: [float(val) for val in values] 
                          for key, values in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        if self.config.verbose >= 1:
            print(f"✓ Model saved to {filepath}")
            print(f"✓ Config saved to {config_path}")
            if self.history is not None:
                print(f"✓ History saved to {history_path}")
    
    def load_model(self, filepath: Path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        filepath : Path
            Path to the saved model file
        """
        self.model = keras.models.load_model(str(filepath))
        
        # Try to load config
        config_path = filepath.parent / f"{filepath.stem}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        if self.config.verbose >= 1:
            print(f"✓ Model loaded from {filepath}")
    
    def get_training_history(self) -> Optional[Dict]:
        """
        Get training history.
        
        Returns:
        --------
        dict: Training history dictionary
        """
        if self.history is None:
            return None
        
        return {key: [float(val) for val in values] 
                for key, values in self.history.history.items()}

