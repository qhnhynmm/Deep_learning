from torch.utils.data import DataLoader, Dataset
from create_data import CreateData

class MyDataset(Dataset):
    def __init__(self, config):
        self.create_data = CreateData(config)
        self.X, self.y = self.create_data()

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self) -> int:
        return len(self.X)

class LoadData:
    def __init__(self, config):
        self.batch_size = config['train']['batch_size']
        self.data = MyDataset(config)
        total_length = len(self.data)
        self.train = int(total_length * 0.8)
        self.dev = int(total_length * 0.1)
        self.test = total_length

    def load_train(self):
        data_train = self.data[:self.train]
        train_loader = DataLoader(dataset=data_train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def load_test(self):
        data_test = self.data[self.train:self.train + self.dev]
        test_loader = DataLoader(dataset=data_test, batch_size=self.batch_size, shuffle=True)
        return test_loader

    def load_dev(self):
        data_dev = self.data[self.train + self.dev:self.train + self.dev + self.test]
        dev_loader = DataLoader(dataset=data_dev, batch_size=self.batch_size, shuffle=True)
        return dev_loader
