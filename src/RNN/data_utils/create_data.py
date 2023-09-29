import torch
from torch import nn
from typing import List, Dict, Optional, Any
import numpy as np
class Create_data(nn.module):
    def __init__(self, config: Dict):
        super(Create_data,self).__init__()
        self.input_size = config['Create_data']['input_size']
        self.output_size = config['Create_data']['output_size']    
        self.sequence_length = config['Create_data']['sequence_length']
        self.num_samples = config['Create_data']['num_samples']
    def create(self):
        X = np.random.randn(self.num_samples, self.sequence_length, self.input_size)
        y = np.random.randn(self.num_samples, self.sequence_length, self.output_size)
        return X, y