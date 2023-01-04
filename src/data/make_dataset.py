# -*- coding: utf-8 -*-
import logging
from glob import glob
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    # Example - Input filepath: "./data/raw/train_data"
    # Example - Output filepath: "./data/processed/train_data"
    # Example - Generate trainset: python3 ./src/data/make_dataset.py "./data/raw/train_data" "./data/processed/train_data"
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Get paths to all images in input_filepath
    filepaths = glob(f"{input_filepath}/*")

    # Create list of tensors with all images and labels
    images = [torch.tensor(np.load(file)["images"]) for file in filepaths]
    labels = [torch.tensor(np.load(file)["labels"]) for file in filepaths]

    # Concatenate the content into a single dataset
    images = torch.cat(images)
    labels = torch.cat(labels)

    # Initial data augmentation
    normalizer = transforms.Normalize((0.5,), (0.5,))
    images = normalizer(images)

    # Save tensor representation
    torch.save(images, f"{output_filepath}/images.pt")
    torch.save(labels, f"{output_filepath}/labels.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
