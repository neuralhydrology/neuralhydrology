import torch
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config


class SHM(BaseConceptualModel):
    """Modified version of SHM [#]_ hydrological model with dynamic parameterization.
    
    The SHM receives the dynamic parameterization given by a deep learning model. This class has two properties which 
    define the initial conditions of the internal states of the model (buckets) and the ranges in which the model 
    parameters are allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R., Azmi, E., and Zehe, E.: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among 
        model elements, Hydrology and Earth System Sciences, 24, 4389–4411, https://doi.org/10.5194/hess-24-4389-2020, 
        2020.
    """

    def __init__(self, cfg: Config):
        super(SHM, self).__init__(cfg=cfg)

    def forward(self, x_conceptual: torch.Tensor,
                lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Performs forward pass on the SHM model. 
        
        In this forward pass, all elements of the batch are processed in  parallel.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)

        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driven model  and will
            be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model
        """
        # get model parameters
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device)
        klu = torch.tensor(0.90, requires_grad=False, dtype=torch.float32,
                           device=x_conceptual.device)  # land use correction factor [-]

        # auxiliary vectors to accelerate the execution of the hydrological model
        t_mean = (x_conceptual[:, :, 2] + x_conceptual[:, :, 3]) / 2
        temp_mask = t_mean < 0
        snow_melt = t_mean * parameters['dd']
        snow_melt[temp_mask] = zero
        # liquid precipitation:
        liquid_p = x_conceptual[:, :, 0].clone()
        liquid_p[temp_mask] = zero
        # solid precipitation (snow):
        snow = x_conceptual[:, :, 0].clone()
        snow[~temp_mask] = zero
        # permanent wilting point use in ET:
        pwp = torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device) * parameters['sumax']

        # Storages
        ss = torch.tensor(self.initial_states['ss'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        sf = torch.tensor(self.initial_states['sf'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        su = torch.tensor(self.initial_states['su'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        si = torch.tensor(self.initial_states['si'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        sb = torch.tensor(self.initial_states['sb'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Snow module --------------------------
            qs_out = torch.minimum(ss, snow_melt[:, j])
            ss = ss - qs_out + snow[:, j]
            qsp_out = qs_out + liquid_p[:, j]

            # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
            qf_in = torch.maximum(zero, qsp_out - parameters['f_thr'][:, j])
            qu_in = torch.minimum(qsp_out, parameters['f_thr'][:, j])

            # Fastflow module ----------------------
            sf = sf + qf_in
            qf_out = sf / parameters['kf'][:, j]
            sf = sf - qf_out

            # Unsaturated zone----------------------
            psi = (su / parameters['sumax'][:, j])**parameters['beta'][:, j]  # [-]
            su_temp = su + qu_in * (1 - psi)
            su = torch.minimum(su_temp, parameters['sumax'][:, j])
            qu_out = qu_in * psi + torch.maximum(zero, su_temp - parameters['sumax'][:, j])  # [mm]
            # Evapotranspiration -------------------
            ktetha = su / parameters['sumax'][:, j]
            et_mask = su <= pwp[:, j]
            ktetha[~et_mask] = one
            ret = x_conceptual[:, j, 1] * klu * ktetha  # [mm]
            su = torch.maximum(zero, su - ret)  # [mm]

            # Interflow reservoir ------------------
            qi_in = qu_out * parameters['perc'][:, j]  # [mm]
            si = si + qi_in  # [mm]
            qi_out = si / parameters['ki'][:, j]  # [mm]
            si = si - qi_out  # [mm]

            # Baseflow reservoir -------------------
            qb_in = qu_out * (1.0 - parameters['perc'][:, j])  # [mm]
            sb = sb + qb_in  # [mm]
            qb_out = sb / parameters['kb'][:, j]  # [mm]
            sb = sb - qb_out

            # Store time evolution of the internal states
            states['ss'][:, j] = ss
            states['sf'][:, j] = sf
            states['su'][:, j] = su
            states['si'][:, j] = si
            states['sb'][:, j] = sb

            # total outflow
            out[:, j, 0] = qf_out + qi_out + qb_out  # [mm]

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {'ss': 0.0, 'sf': 1.0, 'su': 5.0, 'si': 10.0, 'sb': 15.0}

    @property
    def parameter_ranges(self):
        return {
            'dd': [0.0, 10.0],
            'f_thr': [10.0, 60.0],
            'sumax': [20.0, 700.0],
            'beta': [1.0, 6.0],
            'perc': [0.0, 1.0],
            'kf': [1.0, 20.0],
            'ki': [1.0, 100.0],
            'kb': [10.0, 1000.0]
        }
