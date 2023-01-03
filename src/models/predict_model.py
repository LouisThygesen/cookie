# My own project dependencies 
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
    print('Using GPU!')
else:
    device = torch.device("cpu")
    print('Using CPU!')

def get_setup():
    """ Description: Gets hyper-parameters from command line and stores all hyper-parameters in an object
        Return: config-object """

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description='Get hyper-parameters')

    argparser.add_argument('-w', type=str, help='weight file')
    argparser.add_argument('-d', type=str, help='image file')

    args = argparser.parse_args()

    return args

def main():
    # Get experimental setup and create hyperparameter config
    args = get_setup()

    # Load images as np-files and turn into tensor (of correct format)
    img = Image.open(args.m)
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)

    # Initialize model from weights 
    model = Net()
    model.load_state_dict(torch.load(args.w))

    # Run model forward pass on tensor and print result # TODO: Fix input dimenstions
    print(f'Class: {model(img)}')     

if __name__ == '__main__':
    main()




