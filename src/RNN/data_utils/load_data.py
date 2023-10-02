from torch.utils.data import DataLoader, Dataset
from create_data import Create_data
from typing import List, Dict, Optional, Any

class My_data_set(Dataset):
    def __init__(self, config):
        self.create_data = Create_data(config)
        self.X, self.y = self.create_data.create()

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self) -> int:
        return len(self.X)

class Load_data:
    def __init__(self, config):
        self.batch_size = config['train']['batch_size']
        self.data_X, self.data_y = My_data_set(config)
        total_length = len(self.data_X)
        self.train = int(total_length * 0.8)
        self.dev = int(total_length * 0.1)
        self.test = total_length

        self.data_train_X = self.data_X[:self.train]
        self.data_train_y = self.data_y[:self.train]
        self.data_test_X = self.data_X[self.train:self.train + self.dev]
        self.data_test_y = self.data_y[self.train:self.train + self.dev]
        self.data_dev_X = self.data_X[self.train + self.dev:self.train + self.dev + self.test]
        self.data_dev_y = self.data_y[self.train + self.dev:self.train + self.dev + self.test]

    def load_train(self):
        train_dataset = My_data_set(config)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def load_test(self):
        test_dataset = My_data_set(config)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader

    def load_dev(self):
        dev_dataset = My_data_set(config)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=self.batch_size, shuffle=False)
        return dev_loader
