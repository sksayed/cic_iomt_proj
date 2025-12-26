"""
Optuna-based hyperparameter tuning with early stopping (pruning) for CIC IoMT models.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import optuna
from optuna.integration import KerasPruningCallback
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from optuna.samplers import TPESampler
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import json
import pickle
import xgboost as xgb

from ..config.training_config import TrainingConfig
from .trainer import ModelTrainer
from .architectures import ModelBuilder
from ..utils import get_project_paths


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuning using Optuna with early stopping (pruning).
    
    Features:
    - Automatic hyperparameter optimization
    - Early stopping via Optuna pruning
    - Support for multiple architectures
    - Best trial tracking and saving
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        architecture: str = 'dense',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        pruner_type: str = 'median',
        direction: str = 'minimize',
        metric: str = 'val_loss'
    ):
        """
        Initialize Optuna hyperparameter tuner.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        n_classes : int
            Number of output classes
        architecture : str
            Model architecture ('dense', 'cnn_1d', 'deep_dense', 'xgboost')
        n_trials : int
            Number of optimization trials
        timeout : int, optional
            Maximum time in seconds for optimization
        study_name : str, optional
            Name for the Optuna study
        storage : str, optional
            Database URL for study persistence (e.g., 'sqlite:///study.db')
        pruner_type : str
            Pruner type ('median', 'halving', 'hyperband', 'none')
        direction : str
            Optimization direction ('minimize' or 'maximize')
        metric : str
            Metric to optimize ('val_loss', 'val_accuracy', etc.)
        """
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.architecture = architecture
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        
        # Setup paths
        self.paths = get_project_paths()
        self.results_dir = self.paths['RESULTS_DIR'] / 'optuna_studies'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup pruner (following Optuna best practices)
        if pruner_type == 'median':
            # MedianPruner: Good starting point, compares against median performance
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_type == 'halving':
            # SuccessiveHalvingPruner: Allocates more resources to promising trials
            self.pruner = SuccessiveHalvingPruner()
        elif pruner_type == 'hyperband':
            # HyperbandPruner: Often more effective, uses adaptive resource allocation
            self.pruner = HyperbandPruner()
        else:
            self.pruner = None
        
        # Setup study
        study_name = study_name or f"iomt_{architecture}_study"
        sampler = TPESampler(seed=42)  # TPE is default and recommended
        
        # Use SQLite storage by default for persistence (best practice)
        if storage is None:
            storage_path = self.results_dir / f"{study_name}.db"
            storage = f"sqlite:///{storage_path}"
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=True  # Resume study if it exists
        )
        
        self.best_trial: Optional[optuna.Trial] = None
        self.best_model: Optional[keras.Model] = None
        self.best_config: Optional[TrainingConfig] = None
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns:
        --------
        dict: Dictionary of suggested hyperparameters
        """
        params = {}
        
        # Learning rate
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True
        )
        
        # Batch size
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [32, 64, 128, 256]
        )
        
        # Optimizer
        params['optimizer'] = trial.suggest_categorical(
            'optimizer', ['adam', 'sgd', 'rmsprop']
        )
        
        # Architecture-specific parameters
        if self.architecture == 'dense' or self.architecture == 'deep_dense':
            # Number of hidden layers
            n_layers = trial.suggest_int('n_hidden_layers', 2, 5)
            
            # Hidden layer sizes
            hidden_layers = []
            for i in range(n_layers):
                units = trial.suggest_int(
                    f'hidden_units_layer_{i}',
                    32, 512,
                    step=32
                )
                hidden_layers.append(units)
            params['hidden_layers'] = hidden_layers
            
            # Dropout rate
            params['dropout_rate'] = trial.suggest_float(
                'dropout_rate', 0.1, 0.6, step=0.1
            )
            
            # Activation function
            params['activation'] = trial.suggest_categorical(
                'activation', ['relu', 'elu', 'tanh', 'swish']
            )
        
        elif self.architecture == 'cnn_1d':
            # Number of conv layers
            n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
            
            # Filters for each layer
            filters = []
            for i in range(n_conv_layers):
                filter_count = trial.suggest_categorical(
                    f'filters_layer_{i}',
                    [32, 64, 128, 256]
                )
                filters.append(filter_count)
            params['filters'] = filters
            
            # Kernel sizes
            kernel_sizes = []
            for i in range(n_conv_layers):
                kernel_size = trial.suggest_int(
                    f'kernel_size_layer_{i}',
                    3, 7,
                    step=2
                )
                kernel_sizes.append(kernel_size)
            params['kernel_sizes'] = kernel_sizes
            
            # Dropout rate
            params['dropout_rate'] = trial.suggest_float(
                'dropout_rate', 0.1, 0.6, step=0.1
            )
            
            # Activation function
            params['activation'] = trial.suggest_categorical(
                'activation', ['relu', 'elu', 'tanh']
            )
        
        elif self.architecture == 'xgboost':
            # XGBoost-specific hyperparameters
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000, step=50)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 7)
            params['gamma'] = trial.suggest_float('gamma', 0.0, 0.5, step=0.1)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0, step=0.1)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.1)
            params['use_gpu'] = trial.suggest_categorical('use_gpu', [True, False])
            # Learning rate for XGBoost (eta)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            # Early stopping rounds
            params['early_stopping_rounds'] = trial.suggest_int('early_stopping_rounds', 10, 50)
            return params  # XGBoost doesn't use epochs
        
        # Training epochs (can be adjusted per trial) - for neural networks
        params['epochs'] = trial.suggest_int('epochs', 50, 200)
        
        # Early stopping patience
        params['early_stopping_patience'] = trial.suggest_int(
            'early_stopping_patience', 10, 30
        )
        
        return params
    
    def _build_model_from_trial(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any]
    ) -> keras.Model:
        """Build model with trial-specific hyperparameters."""
        # Select optimizer
        if params['optimizer'].lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        elif params['optimizer'].lower() == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=params['learning_rate'],
                momentum=0.9
            )
        elif params['optimizer'].lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=params['learning_rate'])
        else:
            optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        # Build model based on architecture
        if self.architecture == 'dense':
            model = ModelBuilder.build_dense_network(
                input_dim=self.input_dim,
                n_classes=self.n_classes,
                hidden_layers=params['hidden_layers'],
                activation=params['activation'],
                output_activation='softmax',
                dropout_rate=params['dropout_rate']
            )
        elif self.architecture == 'cnn_1d':
            model = ModelBuilder.build_cnn_1d(
                input_dim=self.input_dim,
                n_classes=self.n_classes,
                filters=params['filters'],
                kernel_sizes=params['kernel_sizes'],
                activation=params['activation'],
                output_activation='softmax',
                dropout_rate=params['dropout_rate']
            )
        elif self.architecture == 'deep_dense':
            model = ModelBuilder.build_deep_dense(
                input_dim=self.input_dim,
                n_classes=self.n_classes,
                hidden_layers=params['hidden_layers'],
                activation=params['activation'],
                output_activation='softmax',
                dropout_rate=params['dropout_rate']
            )
        elif self.architecture == 'xgboost':
            # XGBoost models are built in _objective_xgboost
            raise ValueError("XGBoost models should be built in _objective_xgboost")
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        This function is called for each trial and includes early stopping
        via Optuna's pruning mechanism.
        """
        # Route to XGBoost objective if needed
        if self.architecture == 'xgboost':
            return self._objective_xgboost(trial, X_train, y_train, X_val, y_val)
        
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        # Build model
        model = self._build_model_from_trial(trial, params)
        
        # Create callbacks
        callbacks = []
        
        # Optuna pruning callback (early stopping)
        pruning_callback = KerasPruningCallback(
            trial,
            self.metric  # Monitor this metric for pruning
        )
        callbacks.append(pruning_callback)
        
        # Model checkpoint (save best model during trial)
        checkpoint_path = self.results_dir / f"trial_{trial.number}_best.h5"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.metric,
                save_best_only=True,
                verbose=0,
                mode='min' if 'loss' in self.metric else 'max'
            )
        )
        
        # Early stopping (backup, in case pruning doesn't trigger)
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=self.metric,
                patience=params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=0
            )
        )
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0  # Set to 1 for verbose output
        )
        
        # Get best validation metric from history
        best_epoch = np.argmin(history.history[self.metric]) if 'loss' in self.metric else np.argmax(history.history[self.metric])
        best_value = history.history[self.metric][best_epoch]
        
        # Store trial-specific information
        trial.set_user_attr('best_epoch', int(best_epoch))
        trial.set_user_attr('total_epochs', len(history.history['loss']))
        trial.set_user_attr('final_train_loss', float(history.history['loss'][-1]))
        trial.set_user_attr('final_val_loss', float(history.history['val_loss'][-1]))
        trial.set_user_attr('final_train_acc', float(history.history['accuracy'][-1]))
        trial.set_user_attr('final_val_acc', float(history.history['val_accuracy'][-1]))
        
        # Clean up checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        return float(best_value)
    
    def _objective_xgboost(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Objective function for XGBoost optimization with GPU support.
        
        Uses Optuna's built-in XGBoost pruning support.
        """
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        # Build XGBoost model
        model = ModelBuilder.build_xgboost_model(
            n_classes=self.n_classes,
            use_gpu=params['use_gpu'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
        )
        
        # Train model with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=params['early_stopping_rounds'],
            verbose=False
        )
        
        # Get best score from validation set
        # XGBoost stores evaluation results in evals_result_
        eval_results = model.evals_result_
        val_metric_key = list(eval_results['validation_0'].keys())[0]
        val_scores = eval_results['validation_0'][val_metric_key]
        
        # Get best score based on metric type
        # For XGBoost, we use mlogloss (multi-class log loss) which should be minimized
        # If metric is val_loss, we minimize; if val_accuracy, we'd maximize (but XGBoost uses loss)
        if 'loss' in self.metric.lower() or 'mlogloss' in val_metric_key:
            best_score = min(val_scores)
        else:
            # For accuracy metrics, maximize
            best_score = max(val_scores)
        
        # Report intermediate value for pruning
        trial.report(best_score, step=len(val_scores) - 1)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Store trial-specific information
        trial.set_user_attr('n_estimators_used', model.get_booster().num_boosted_rounds())
        trial.set_user_attr('use_gpu', params['use_gpu'])
        
        return float(best_score)
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        show_progress: bool = True
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        optuna.Study: Completed study with optimization results
        """
        print("=" * 80)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"Architecture: {self.architecture}")
        if self.architecture == 'xgboost':
            print("GPU Acceleration: Enabled (if GPU available)")
        print(f"Number of trials: {self.n_trials}")
        print(f"Optimization metric: {self.metric}")
        print(f"Direction: {self.direction}")
        print(f"Pruner: {self.pruner.__class__.__name__ if self.pruner else 'None'}")
        print(f"Early stopping: Enabled via Optuna Pruning")
        print(f"Study storage: {self.study.storage if hasattr(self.study, 'storage') else 'In-memory'}")
        print("=" * 80)
        
        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress
        )
        
        # Store best trial
        self.best_trial = self.study.best_trial
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Best trial number: {self.best_trial.number}")
        print(f"Best {self.metric}: {self.best_trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in self.best_trial.params.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
        return self.study
    
    def train_best_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None
    ) -> Union[Tuple[keras.Model, keras.callbacks.History], Tuple[xgb.XGBClassifier, Dict]]:
        """
        Train the best model found during optimization.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        epochs : int, optional
            Number of epochs (uses best trial's epochs if None, ignored for XGBoost)
            
        Returns:
        --------
        tuple: (trained_model, training_history) - Keras models return History, XGBoost returns dict
        """
        if self.best_trial is None:
            raise ValueError("No optimization has been run. Call optimize() first.")
        
        print("\n" + "=" * 80)
        print("TRAINING BEST MODEL")
        print("=" * 80)
        
        # Handle XGBoost separately
        if self.architecture == 'xgboost':
            return self._train_best_xgboost(X_train, y_train, X_val, y_val)
        
        # Get best hyperparameters
        params = self.best_trial.params.copy()
        if epochs is not None:
            params['epochs'] = epochs
        
        # Build model with best hyperparameters
        model = self._build_model_from_trial(self.best_trial, params)
        
        # Create training config
        config = TrainingConfig(
            input_dim=self.input_dim,
            n_classes=self.n_classes,
            hidden_layers=params.get('hidden_layers', [128, 64, 32]),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.3),
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer'],
            early_stopping_patience=params.get('early_stopping_patience', 15),
            monitor_metric=self.metric
        )
        
        self.best_config = config
        
        # Create callbacks
        callbacks = []
        
        # Model checkpoint
        best_model_path = self.paths['MODELS_DIR'] / 'best_optuna_model.h5'
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor=self.metric,
                save_best_only=True,
                verbose=1,
                mode='min' if 'loss' in self.metric else 'max'
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=self.metric,
                patience=params.get('early_stopping_patience', 15),
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=self.metric,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.best_model = model
        
        print(f"\n✓ Best model trained and saved to: {best_model_path}")
        print("=" * 80)
        
        return model, history
    
    def _train_best_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """Train best XGBoost model with optimal hyperparameters."""
        params = self.best_trial.params.copy()
        
        # Build XGBoost model with best hyperparameters
        model = ModelBuilder.build_xgboost_model(
            n_classes=self.n_classes,
            use_gpu=params['use_gpu'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
        )
        
        # Train model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=params['early_stopping_rounds'],
            verbose=True
        )
        
        # Get evaluation results as history dict
        eval_results = model.evals_result_
        history = {
            'train': eval_results.get('validation_0', {}),
            'val': eval_results.get('validation_0', {})
        }
        
        # Save model
        best_model_path = self.paths['MODELS_DIR'] / 'best_optuna_xgboost_model.json'
        model.save_model(str(best_model_path))
        
        self.best_model = model
        
        print(f"\n✓ Best XGBoost model trained and saved to: {best_model_path}")
        if params['use_gpu']:
            print("✓ GPU acceleration was enabled")
        print("=" * 80)
        
        return model, history
    
    def save_study(self, filename: Optional[str] = None):
        """Save study results to file."""
        if filename is None:
            filename = f"optuna_study_{self.architecture}_{self.study.study_name}.json"
        
        filepath = self.results_dir / filename
        
        # Save study summary
        study_summary = {
            'study_name': self.study.study_name,
            'n_trials': len(self.study.trials),
            'best_trial_number': self.best_trial.number if self.best_trial else None,
            'best_value': self.best_trial.value if self.best_trial else None,
            'best_params': self.best_trial.params if self.best_trial else None,
            'direction': self.direction,
            'metric': self.metric,
            'architecture': self.architecture
        }
        
        with open(filepath, 'w') as f:
            json.dump(study_summary, f, indent=2)
        
        print(f"✓ Study saved to: {filepath}")
        
        return filepath
    
    def get_trial_summary(self) -> Dict[str, Any]:
        """Get summary of all trials."""
        if len(self.study.trials) == 0:
            return {}
        
        summary = {
            'n_trials': len(self.study.trials),
            'n_complete': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_trial': {
                'number': self.best_trial.number if self.best_trial else None,
                'value': self.best_trial.value if self.best_trial else None,
                'params': self.best_trial.params if self.best_trial else None
            }
        }
        
        return summary

