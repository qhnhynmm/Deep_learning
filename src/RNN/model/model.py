
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class SimpleRNN_Model(nn.Module):
    def __init__(self, config):
        super(SimpleRNN_Model, self).__init__()
        self.rnn = SimpleRNN(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        if labels is not None:
            logits = self.rnn(inputs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.rnn(inputs)
            return logits
