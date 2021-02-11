class NoTrainDataError(Exception):
    """Raised, when basin contains no valid discharge in training period"""


class AllNaNError(Exception):
    """Raised by `calculate_(all_)metrics` if all observations or all simulations are NaN. """
