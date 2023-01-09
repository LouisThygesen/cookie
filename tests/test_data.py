# NEEDS TO BE TESTED

import torch
import os.path
import pytest

from src.data.dataset_class import mnist

class TestData():
    def test_one(self):
        dataset = mnist('data/processed/train_data')
        assert len(dataset) == 40000, "wrong number of obs in train set"

    def test_two(self):
        dataset = mnist('data/processed/test_data')
        assert len(dataset) == 5000, "wrong number of obs in test set"

    def test_three(self):
        dataset = mnist('data/processed/train_data')
        img, _ = dataset.__getitem__(0)
        
        print(type(img.shape))
        assert list(img.shape) == [28,28], "shape check" 

    def test_four(self):
        dataset = mnist('data/processed/train_data')
        assert list(torch.unique(dataset.labels)) == [0,1,2,3,4,5,6,7,8,9], "label check"




    
