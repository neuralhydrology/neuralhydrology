from typing import Sequence

import numpy as np


class RatingCurve(object):
    """Class to estimate rating curves from stage and discharge data and to convert stage to discharge and vice-versa.
            
    Parameters
    ----------
    stages : Sequence
        Stage data to estimate a rating curve 
    discharges : Sequence
        Discharge data to estimate a rating curve 
    move_stages_to_zero : bool, optional
        Option to account for any offset in the stage data. This will automatically set the minimum measured stage to 
        zero.
    """

    def __init__(self, stages: Sequence[float], discharges: Sequence[float], move_stages_to_zero: bool = True):

        # Validate input.
        if len(stages) != len(discharges):
            raise ValueError("The sequence 'stages' and 'discharges' must have the same length")

        self.stages = np.array(stages, dtype=np.float32)
        self.discharges = np.array(discharges, dtype=np.float32)

        # In the interpolation, x-axis is stage and y-axis is discharge.
        if move_stages_to_zero:
            use_stages = self.stages - np.min(self.stages)
            self.zero_stages = np.min(self.stages)
        else:
            use_stages = self.stages
            self.zero_stages = False

        self.coeffs = np.polyfit(use_stages, self.discharges, deg=2)

    def stage_to_discharge(self, stage: np.ndarray) -> np.ndarray:
        """Convert stage to discharge.
            
        Parameters
        ----------
        stage : np.ndarray
            Stage data to convert to discharge 
    
        Returns
        -------
        np.ndarray
            Estimated discharge 
        """

        if self.zero_stages:
            stage -= self.zero_stages

        return self.coeffs[2] + stage * self.coeffs[1] + stage**2 * self.coeffs[0]

    def discharge_to_stage(self, discharge: np.ndarray) -> np.ndarray:
        """Convert discharge to stage.
            
        Parameters
        ----------
        discharge : np.ndarray
            Discharge data to convert to stage 
  
        Returns
        -------
        np.ndarray
            Estimated stage 
        """

        # init storage
        stage = np.full(discharge.shape, np.nan)
        solutions = np.full([discharge.shape[0], 2], np.nan)

        centered_bias = self.coeffs[2] - discharge
        radicand = self.coeffs[1]**2 - 4 * self.coeffs[0] * centered_bias
        radicand[np.isnan(radicand)] = -9999
        mask = radicand >= 0

        solutions[mask, 0] = (-self.coeffs[1] + np.sqrt(radicand[mask])) / (2 * self.coeffs[0])
        solutions[mask, 1] = (-self.coeffs[1] - np.sqrt(radicand[mask])) / (2 * self.coeffs[0])

        best_idx = np.argmin(np.abs(solutions[mask] - np.expand_dims(discharge[mask], -1)), axis=1)

        stage[mask] = solutions[mask, best_idx]

        if self.zero_stages:
            stage += self.zero_stages

        return stage
