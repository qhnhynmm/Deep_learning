import torch
from torch import nn

class CreateData(nn.Module):
    def __init__(self, config):
        super(CreateData, self).__init__()
        self.input_size = config['Create_data']['input_size']
        self.output_size = config['Create_data']['output_size']
        self.sequence_length = config['Create_data']['sequence_length']
        self.num_samples = config['Create_data']['num_samples']

    def forward(self):
        X = torch.randn(self.num_samples, self.sequence_length, self.input_size)
        y = torch.randn(self.num_samples, self.sequence_length, self.output_size)
        return X, y
