# My own project dependencies 
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset_class import mnist
from src.models.model import Net

# Set up GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU!')
else:
    device = torch.device("cpu")
    print('Using CPU!')

def get_setup(): # TODO: Is this still relevant?
    """ Description: Gets hyper-parameters from command line and stores all hyper-parameters in an object
        Return: config-object """

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description='Get hyper-parameters')
    argparser.add_argument('-lr', type=float, default="1e-3", help='learning rate')

    args = argparser.parse_args()

    return args

class hyperparameters():
    def __init__(self, args):
        self.batch_size = 64
        self.epochs = 10
        self.lr = args.lr

def train_epoch(model, criterion, optimizer, train_dl):
    model.train()
    train_batch_losses = []

    for i, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Save running loss
        train_batch_losses.append(loss.item())

    return np.mean(train_batch_losses)

def eval_epoch(model, criterion, eval_dl):
    model.eval()
    eval_batch_losses = []

    for i, (imgs, labels) in enumerate(eval_dl):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs, labels)
        loss = criterion(outputs, labels)

        # Save evaluation loss
        eval_batch_losses.append(loss.item())

    return np.mean(eval_batch_losses)

def train(config, model, criterion, optimizer, train_dl, eval_dl):
    train_epoch_loss = []
    eval_epoch_loss = []
    best_loss = float('inf')

    # Create directory to save best model
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    weights_path = './models/{}'.format(current_time)
    os.mkdir(weights_path)

    for epoch in range(config.epochs):
        train_batch_loss = train_epoch(config, model, criterion, optimizer, train_dl)
        eval_batch_loss = eval_epoch(model, criterion, eval_dl)

        # Save losses
        train_epoch_loss.append(train_batch_loss)
        eval_epoch_loss.append(eval_batch_loss)

        # Save if model is better
        if eval_batch_loss < best_loss:
            best_loss = eval_batch_loss
            torch.save(model.state_dict(), f'{weights_path}/model.pth')

    # Plot grap of training and evaluation loss
    plt.figure()
    plt.plot(train_epoch_loss, label='Training loss')
    plt.plot(eval_epoch_loss, label='Evaluation loss')
    plt.legend()
    plt.savefig('./reports/figures/loss.png')

def main():
    # Get experimental setup and create hyperparameter config
    args = get_setup()
    config = hyperparameters(args)

    # Initialize dataset (from preprocessed data) # TODO: Add reference to preprocessed data 
    train_set = mnist('')
    eval_set = mnist('')

    # Initialize dataloader
    train_dl = DataLoader(train_set, batch_size=64, shuffle=True)
    eval_dl = DataLoader(eval_set, batch_size=64, shuffle=False)

    # Initialize model
    model = Net().to(device)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)

    # Train or evaluate model
    train(config, model, criterion, optimizer, train_dl, eval_dl)

if __name__ == '__main__':
    main()