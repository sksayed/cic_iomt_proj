# Model Training Module

This module provides class-based training functionality for the CIC IoMT dataset.

## Structure

- `architectures.py`: Model architecture builders (Dense, CNN-1D, Deep Dense)
- `trainer.py`: Main training class with training, evaluation, and model management

## Usage

### Basic Example

```python
from config.training_config import TrainingConfig
from models.trainer import ModelTrainer
from utils import get_project_paths
import numpy as np
import pickle

# Load preprocessed data
paths = get_project_paths()
X_train = np.load(paths['OUTPUT_DIR'] / 'X_train.npy')
y_train = np.load(paths['OUTPUT_DIR'] / 'y_train.npy')
X_val = np.load(paths['OUTPUT_DIR'] / 'X_val.npy')
y_val = np.load(paths['OUTPUT_DIR'] / 'y_val.npy')

# Load preprocessing info
with open(paths['MODELS_DIR'] / 'preprocessing_info.pkl', 'rb') as f:
    preprocess_info = pickle.load(f)

# Create config
config = TrainingConfig(
    input_dim=preprocess_info['n_features'],
    n_classes=preprocess_info['n_classes'],
    hidden_layers=[256, 128, 64],
    batch_size=64,
    epochs=100,
    learning_rate=0.001
)

# Train model
trainer = ModelTrainer(config)
trainer.build_model(architecture='dense')
history = trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
X_test = np.load(paths['OUTPUT_DIR'] / 'X_test.npy')
y_test = np.load(paths['OUTPUT_DIR'] / 'y_test.npy')
metrics = trainer.evaluate(X_test, y_test)

# Save model
trainer.save_model()
```

## Available Architectures

- `'dense'`: Standard dense (fully connected) network
- `'cnn_1d'`: 1D Convolutional Neural Network
- `'deep_dense'`: Deeper dense network with more layers

## Configuration Options

See `config/training_config.py` for all available configuration options including:
- Model architecture parameters
- Training hyperparameters
- Callback settings
- Model saving options

