"""
Model architecture definitions for CIC IoMT dataset.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional, Dict, Any
import xgboost as xgb


class ModelBuilder:
    """Builder class for creating neural network architectures."""
    
    @staticmethod
    def build_dense_network(
        input_dim: int,
        n_classes: int,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        output_activation: str = 'softmax',
        dropout_rate: float = 0.3,
        name: str = 'dense_classifier'
    ) -> keras.Model:
        """
        Build a dense (fully connected) neural network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        n_classes : int
            Number of output classes
        hidden_layers : list
            List of hidden layer sizes
        activation : str
            Activation function for hidden layers
        output_activation : str
            Activation function for output layer
        dropout_rate : float
            Dropout rate for regularization
        name : str
            Model name
        
        Returns:
        --------
        keras.Model: Uncompiled model
        """
        model = keras.Sequential(name=name)
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(n_classes, activation=output_activation))
        
        return model
    
    @staticmethod
    def build_cnn_1d(
        input_dim: int,
        n_classes: int,
        filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3],
        activation: str = 'relu',
        output_activation: str = 'softmax',
        dropout_rate: float = 0.3,
        name: str = 'cnn_1d_classifier'
    ) -> keras.Model:
        """
        Build a 1D CNN for sequence-like data.
        
        Note: Reshapes input to (batch, input_dim, 1) for CNN processing.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        n_classes : int
            Number of output classes
        filters : list
            List of filter counts for each conv layer
        kernel_sizes : list
            List of kernel sizes for each conv layer
        activation : str
            Activation function for hidden layers
        output_activation : str
            Activation function for output layer
        dropout_rate : float
            Dropout rate for regularization
        name : str
            Model name
        
        Returns:
        --------
        keras.Model: Uncompiled model
        """
        model = keras.Sequential(name=name)
        
        # Reshape input for CNN
        model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
        
        # Convolutional layers
        for i, (filters_count, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            model.add(layers.Conv1D(filters_count, kernel_size, activation=activation))
            model.add(layers.BatchNormalization())
            if i < len(filters) - 1:  # No pooling after last conv layer
                model.add(layers.MaxPooling1D(2))
        
        # Global pooling and dense layers
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(64, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(n_classes, activation=output_activation))
        
        return model
    
    @staticmethod
    def build_deep_dense(
        input_dim: int,
        n_classes: int,
        hidden_layers: List[int] = [512, 256, 128, 64],
        activation: str = 'relu',
        output_activation: str = 'softmax',
        dropout_rate: float = 0.3,
        name: str = 'deep_dense_classifier'
    ) -> keras.Model:
        """
        Build a deeper dense network with more layers.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        n_classes : int
            Number of output classes
        hidden_layers : list
            List of hidden layer sizes
        activation : str
            Activation function for hidden layers
        output_activation : str
            Activation function for output layer
        dropout_rate : float
            Dropout rate for regularization
        name : str
            Model name
        
        Returns:
        --------
        keras.Model: Uncompiled model
        """
        model = keras.Sequential(name=name)
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with batch normalization
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(n_classes, activation=output_activation))
        
        return model
    
    @staticmethod
    def build_xgboost_model(
        n_classes: int,
        use_gpu: bool = True,
        tree_method: str = 'hist',
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Build an XGBoost classifier with GPU acceleration support.
        
        Parameters:
        -----------
        n_classes : int
            Number of output classes
        use_gpu : bool
            Enable GPU acceleration (requires CUDA)
        tree_method : str
            Tree construction method ('hist' for GPU, 'approx' for CPU)
        **kwargs : dict
            Additional XGBoost parameters
        
        Returns:
        --------
        xgb.XGBClassifier: XGBoost classifier model
        """
        # Base parameters
        params = {
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'num_class': n_classes if n_classes > 2 else None,
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'random_state': 42,
            'verbosity': 0,  # Suppress XGBoost output
        }
        
        # GPU acceleration settings
        if use_gpu:
            params.update({
                'tree_method': 'hist',  # GPU-accelerated histogram method
                'device': 'cuda',  # Use GPU
                'gpu_id': 0,  # Use first GPU
            })
        else:
            params.update({
                'tree_method': tree_method,
                'device': 'cpu',
            })
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        model = xgb.XGBClassifier(**params)
        
        return model

