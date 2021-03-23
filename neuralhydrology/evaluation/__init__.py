from pathlib import Path

from neuralhydrology.evaluation.tester import BaseTester, RegressionTester, UncertaintyTester
from neuralhydrology.utils.config import Config


def get_tester(cfg: Config, run_dir: Path, period: str, init_model: bool) -> BaseTester:
    """Get specific tester class objects depending on the model (head) type.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.

    Returns
    -------
    BaseTester
        `RegressionTester` if the model head is 'regression'. `UncertaintyTester` if the model head is one of 
        {'gmm', 'cmal', 'umal'} or if the evaluation is run in MC-Dropout mode.
    """
    if cfg.mc_dropout or cfg.head.lower() in ["gmm", "cmal", "umal"]:
        Tester = UncertaintyTester
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    elif cfg.head.lower() in ["regression", ""]:
        Tester = RegressionTester
    else:
        NotImplementedError(f"No evaluation method implemented for {cfg.head} head")

    return Tester(cfg=cfg, run_dir=run_dir, period=period, init_model=init_model)
