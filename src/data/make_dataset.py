# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
# My own project dependencies 
import numpy as np
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from torchvision import transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # List of content from each training npz-file
    images = [np.load(file)['images'] for file in input_filepath]
    labels = [np.load(file)['labels'] for file in input_filepath]

    # Concatenate the content into a single dataset
    imgs = np.concatenate(images)
    labels = np.concatenate(labels)

    # Transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    imgs = transform(imgs)

    # Save as npz file
    np.savez(output_filepath, images=imgs, labels=labels)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
