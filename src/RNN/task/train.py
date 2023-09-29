import torch
import torch.nn as nn
import torch.optim as optim
import os
from model.build_model import CustomRNN
class RNN_task:
    def __init__(self, config):
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.batch_size = config['train']['batch_size']
        self.num_epochs = config['train']['num_epochs']
        self.early_stopping_patience = config['train']['early_stopping_patience']
        self.learning_rate = config['train']['learning_rate']
        self.hidden_size = config['train']['hidden_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Định nghĩa mô hình RNN
        self.model = self.CustomRNN(self.input_size, self.hidden_size, self.output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    def training(self, X_train, y_train):
        patience_counter = 0
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size].to(self.device)
                y_batch = y_train[i:i + self.batch_size].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.view(-1, self.output_size), y_batch.view(-1, self.output_size))
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

            # Early Stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                # Lưu checkpoint
                torch.save(self.model.state_dict(), 'rnn_checkpoint.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping...")
                    break

        # Load mô hình tốt nhất từ checkpoint
        self.model.load_state_dict(torch.load('rnn_checkpoint.pth'))
        self.model.eval()
    def forward_one_step(self, x):
        with torch.no_grad():
            h = torch.zeros(1, self.hidden_size, dtype=torch.float32, device=self.device)
            h = h.view(1, 1, -1)
            x = x.view(1, 1, -1)
            out, _ = self.model(x, h)
            return out.view(1, -1).cpu().numpy()

rnn_task = RNN_task(config)

# Huấn luyện mô hình
rnn_task.training(X_train, y_train)

# Dự đoán đầu ra cho một ví dụ
test_input = torch.randn(sequence_length, config['input_size'])
predicted_output = rnn_task.forward_one_step(test_input)
print("Predicted Output:")
print(predicted_output)
