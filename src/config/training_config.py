"""
Training configuration class for model hyperparameters and settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class TrainingConfig:
    """Configuration class for model training."""
    
    # Model architecture
    input_dim: int
    n_classes: int
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = 'relu'
    output_activation: str = 'softmax'
    dropout_rate: float = 0.3
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    
    # Data
    validation_split: float = 0.2
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Model saving
    save_best_only: bool = True
    monitor_metric: str = 'val_loss'
    model_save_path: Optional[Path] = None
    
    # Logging
    tensorboard_log_dir: Optional[Path] = None
    verbose: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_dim': self.input_dim,
            'n_classes': self.n_classes,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
            'reduce_lr_factor': self.reduce_lr_factor,
            'min_lr': self.min_lr,
            'save_best_only': self.save_best_only,
            'monitor_metric': self.monitor_metric,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Handle Path objects
        if 'model_save_path' in config_dict and config_dict['model_save_path']:
            config_dict['model_save_path'] = Path(config_dict['model_save_path'])
        if 'tensorboard_log_dir' in config_dict and config_dict['tensorboard_log_dir']:
            config_dict['tensorboard_log_dir'] = Path(config_dict['tensorboard_log_dir'])
        
        return cls(**config_dict)

