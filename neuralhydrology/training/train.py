from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config


def start_training(cfg: Config):
    """Start model training.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    """
    if cfg.head.lower() in ['regression']:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    trainer.initialize_training()
    trainer.train_and_validate()
