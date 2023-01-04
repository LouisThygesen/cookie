import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.data.dataset_class import mnist
from src.models.model import Net
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set up GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU!")


def get_setup():
    """Description: Gets hyper-parameters from command line and stores all hyper-parameters in an object
    Return: config-object"""

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description="Get hyper-parameters")
    argparser.add_argument("-lr", type=float, default="1e-3", help="learning rate")

    args = argparser.parse_args()

    return args


class hyperparameters:
    def __init__(self, args):
        self.batch_size = 64
        self.epochs = 2
        self.lr = args.lr


def train_epoch(model, criterion, optimizer, train_dl, epoch):
    model.train()
    train_batch_losses = []

    with tqdm(train_dl) as tepoch:
        for imgs, labels in tepoch:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.unsqueeze(1)

            optimizer.zero_grad()

            outputs, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Save running loss
        train_batch_losses.append(loss.item())

    return np.mean(train_batch_losses)


def train(config, model, criterion, optimizer, train_dl):
    train_epoch_loss = []
    best_loss = float("inf")

    # Create directory to save best model
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    weights_path = "./models/{}".format(current_time)
    os.mkdir(weights_path)

    for epoch in range(config.epochs):
        train_batch_loss = train_epoch(model, criterion, optimizer, train_dl, epoch)
        train_epoch_loss.append(train_batch_loss)

        # Save if model is better
        if train_batch_loss < best_loss:
            best_loss = train_batch_loss
            torch.save(model.state_dict(), f"{weights_path}/model.pth")

    # Plot grap of training  loss
    plt.figure()
    plt.plot(train_epoch_loss, label="Training loss")
    plt.legend()
    plt.savefig("./reports/figures/loss.png")


def main():
    # Get experimental setup and create hyperparameter config
    args = get_setup()
    config = hyperparameters(args)

    # Initialize dataset and dataloader
    train_set = mnist("./data/processed/train_data")
    train_dl = DataLoader(train_set, batch_size=64, shuffle=True)

    # Initialize model
    model = Net().to(device)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)

    # Train or evaluate model
    train(config, model, criterion, optimizer, train_dl)


if __name__ == "__main__":
    main()
