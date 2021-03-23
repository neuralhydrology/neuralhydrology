from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.umaltrainer import UMALTrainer
from neuralhydrology.utils.config import Config


def start_training(cfg: Config):
    """Start model training.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    """
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    if cfg.head.lower() in ['regression', 'gmm', 'cmal', '']:
        trainer = BaseTrainer(cfg=cfg)
    elif cfg.head.lower() == 'umal':
        trainer = UMALTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    trainer.initialize_training()
    trainer.train_and_validate()
