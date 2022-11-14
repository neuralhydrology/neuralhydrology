class NoTrainDataError(Exception):
    """Raised, when basin contains no valid samples in training period"""

class NoEvaluationDataError(Exception):
    """Raised, when basin contains no valid samples in validation or test period""" 

class AllNaNError(Exception):
    """Raised by `calculate_(all_)metrics` if all observations or all simulations are NaN. """
