# definition of temporal neural network. details can be found https://doi.org/10.3389/frwa.2020.00028

from typing import Dict

import torch
from torch import nn
from torch.nn.utils import weight_norm

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer


class Chomp1d(nn.Module):  # causal padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernal_size, stride, dilation, padding, dropout=0.4):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernal_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernal_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # res connection
        self.relu = nn.ReLU()
        # self.init_weights()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # res connection

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


class TCNN(BaseModel):
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['tcnn', 'dense1']

    def __init__(self, cfg: Dict):
        super(TCNN, self).__init__(cfg=cfg)
        self.kernal_size = cfg["kernal_size"]
        self.num_levels = cfg["num_levels"]
        self.num_channels = cfg["num_channels"]
        self.dr_rate = 0.4
        self.embedding_net = InputLayer(cfg)
        n_attributes = 0
        if ("camels_attributes" in cfg.keys()) and cfg["camels_attributes"]:
            print('input attributes')
            n_attributes += len(cfg["camels_attributes"])

        self.input_size = len(cfg["dynamic_inputs"] + cfg.get("static_inputs", [])) + n_attributes
        if cfg["use_basin_id_encoding"]:
            self.input_size += cfg["number_of_basins"]

        layers = []
        # num_levels = len(num_channels) # number of blocks. Should be 2-3. maybe more?

        for i in range(self.num_levels):
            # dilation_size = 2 ** i # dilation rate with layer number
            dilation_size = 6 * (i + 1)
            in_channels = self.input_size if i == 0 else self.num_channels
            out_channels = self.num_channels
            layers += [
                TemporalBlock(in_channels, out_channels, padding=(self.kernal_size - 1) * dilation_size, stride=1,
                              dilation=dilation_size,
                              dropout=self.dr_rate, kernal_size=self.kernal_size)]

        self.tcnn = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=cfg["output_dropout"])

        # self.reset_parameters()
        self.dense1 = nn.Linear(self.num_channels * 20, 100)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(100, 1)
        self.flat = nn.Flatten()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_d = self.embedding_net(data)  # [seq_length, batch_size, n_features]
        ## convert to CNN inputs:
        x_d = x_d.transpose(0, 1)
        x_d = x_d.transpose(1, 2)  # [batch_size, n_features, seq_length]
        tcnn_out = self.tcnn(input=x_d)
        ## slice:
        tcnn_out = tcnn_out[:, :, -20:]

        y_hat = self.dense2(self.dropout(self.act(self.dense1(self.flat(tcnn_out)))))

        # y_hat = y_hat.unsqueeze(1)
        pred = {'y_hat': y_hat}

        return pred  # keep the same form with LSTM's other two outputs
