import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, config):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = config["train"]['hidden_dim']
        self.input_size = config["Create_data"]['input_size']
        self.output_size = config["Create_data"]['output_size']
        
        # Định nghĩa lớp RNN
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        # Đầu vào x có kích thước (batch_size, sequence_length, input_size)
        # Đầu ra sau lớp RNN có kích thước (batch_size, sequence_length, hidden_dim)
        rnn_out, _ = self.rnn(x)
        
        # Lấy đầu ra ở thời điểm cuối cùng trong chuỗi đầu ra của RNN
        output = self.fc(rnn_out[:, -1, :])
        output = torch.softmax(output, dim=1)  # Áp dụng hàm softmax để nhận được xác suất dự đoán
        return output

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
