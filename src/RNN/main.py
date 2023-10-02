import argparse
import torch
from utils.get_config import get_config
from task import train

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()

config = get_config(args.config_file)

torch.manual_seed(1234)

if config.task == 'RNN':
    task = train(config)
    train_loader = task.load_train_data()  # You need a method to load your training data
    dev_loader = task.load_dev_data()  # You need a method to load your validation data
    
    for epoch in range(config.num_epochs):
        # Training
        task.training(train_loader)
        
        # Evaluation
        task.evaluate(dev_loader)
