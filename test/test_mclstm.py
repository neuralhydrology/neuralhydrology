import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.mclstm import MCLSTM


def test_mass_conservation():
    torch.manual_seed(111)

    # create minimal config required for model initialization
    config = Config({
        'dynamic_inputs': ['tmin(C)', 'tmax(C)'],
        'hidden_size': 10,
        'initial_forget_bias': 0,
        'mass_inputs': ['prcp(mm/day)'],
        'model': 'mclstm',
        'target_variables': ['QObs(mm/d)']
    })
    model = MCLSTM(config)

    # create random inputs
    data = {
        'x_d':
            torch.rand((1, 25, 3))  # [batch size, sequence length, total number of time series inputs]
    }

    # get model outputs and intermediate states
    output = model(data)

    # the total mass within the system at each time step is the cumsum over the outgoing mass + the current cell state
    cumsum_system = output["m_out"].sum(-1).cumsum(-1) + output["c"].sum(-1)

    # the accumulated mass of the inputs at each time step
    cumsum_input = data["x_d"][:, :, 0].cumsum(-1)

    # check if the total mass is conserved at every timestep of the forward pass
    assert torch.allclose(cumsum_system, cumsum_input)
