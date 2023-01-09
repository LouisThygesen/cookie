import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.dataset_class import mnist
from src.models.model import Net

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

    argparser.add_argument(
        "-w", type=str, default="./models/18-37-55/model.pth", help="weight file"
    )
    argparser.add_argument(
        "-d", type=str, default="./data/raw/test_data/test.npz", help="image npz file"
    )

    args = argparser.parse_args()

    return args


def main():
    # Get experimental setup and create hyperparameter config
    args = get_setup()

    # Load images as np-files and turn into tensor (of correct format)
    images = torch.tensor(np.load(args.d)["images"])
    images = images.unsqueeze(1).float()

    # Initial data augmentation (normalization)
    normalizer = transforms.Normalize((0.5,), (0.5,))
    images = normalizer(images)

    # Initialize model from weights
    model = Net()
    model.load_state_dict(torch.load(args.w))

    # Run model forward pass on tensor and print 5 first results
    outputs, _ = model(images)
    pred = torch.max(outputs, 1)[1].tolist()
    print(f"First 10 observation classes: {pred[:5]}")


if __name__ == "__main__":
    main()
