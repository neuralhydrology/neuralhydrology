import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility for training with plateau detection.
    
    Monitors a specified metric and stops training when no improvement is observed
    for a specified number of epochs (patience). Supports both minimization and
    maximization of metrics.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping when no improvement is observed.
    min_delta : float, optional
        Minimum change in the metric to qualify as an improvement. Default: 0.0.
    mode : str, optional
        Whether to minimize ('min') or maximize ('max') the metric. Default: 'min'.
    """
    
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        
        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # Initialize best value based on mode
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')
        
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, metric_value: float, epoch: int) -> bool:
        """Check if training should be stopped.
        
        Parameters
        ----------
        metric_value : float
            Current value of the monitored metric.
        epoch : int
            Current epoch number.
            
        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        improved = False
        
        if self.mode == 'min':
            improved = metric_value < self.best_value - self.min_delta
        else:  # mode == 'max'
            improved = metric_value > self.best_value + self.min_delta
        
        if improved:
            # Improvement detected
            self.best_value = metric_value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                LOGGER.info(f"Early stopping triggered at epoch {epoch}. "
                           f"Best metric value: {self.best_value:.5f} at epoch {self.best_epoch}")
                return True
            else:
                LOGGER.debug(f"No improvement for {self.counter}/{self.patience} epochs. "
                            f"Current value: {metric_value:.5f}, best: {self.best_value:.5f}")
                return False
    
    def state_dict(self) -> dict:
        """Return current state for checkpoint saving."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'best_value': self.best_value,
            'counter': self.counter,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.best_value = state_dict['best_value']
        self.counter = state_dict['counter']
        self.best_epoch = state_dict['best_epoch']