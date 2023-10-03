import torch
import torch.nn as nn
import torch.optim as optim
from model.model import SimpleRNN  # Import mô hình SimpleRNN từ module model
from data_utils.load_data import LoadData
class RNN_task:
    def __init__(self, config):
        self.input_size = config['Create_data']['input_size']
        self.output_size = config['Create_data']['output_size']
        self.batch_size = config['train']['batch_size']
        self.num_epochs = config['train']['num_epochs']
        self.early_stopping_patience = config['train']['early_stopping_patience']
        self.learning_rate = config['train']['learning_rate']
        self.hidden_size = config['train']['hidden_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = LoadData(config)
        # Định nghĩa mô hình RNN (sử dụng SimpleRNN từ module model)
        self.model = SimpleRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training(self):
        X_train,y_train = self.data.load_train()
        X_dev,y_dev = self.data.load_dev()
        # X_test,y_test = self.data.load_test()
        patience_counter = 0
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size].to(self.device)
                y_batch = y_train[i:i + self.batch_size].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.view(-1, self.output_size), y_batch.view(-1, self.output_size))
                loss.backward()
                self.optimizer.step()

            # Đánh giá mô hình trên tập dev
            self.model.eval()
            with torch.no_grad():
                dev_loss = 0
                for i in range(0, len(X_dev), self.batch_size):
                    X_batch_dev = X_dev[i:i + self.batch_size].to(self.device)
                    y_batch_dev = y_dev[i:i + self.batch_size].to(self.device)
                    dev_outputs = self.model(X_batch_dev)
                    dev_loss += self.criterion(dev_outputs.view(-1, self.output_size), y_batch_dev.view(-1, self.output_size)).item()
                dev_loss /= len(X_dev) / self.batch_size

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {loss.item():.4f}, Dev Loss: {dev_loss:.4f}')

                # Lưu checkpoint nếu có kết quả tốt nhất trên tập dev
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    patience_counter = 0
                    # Lưu checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss
                    }, 'best_rnn_checkpoint.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print("Early stopping...")
                        break