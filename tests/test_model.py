# NEEDS TO BE TESTED

import pytest
import torch

from src.models.model import Net


class TestModel:
    def test_one(self):
        x = torch.rand([1, 28, 28])
        model = Net()
        pred, _ = model(x)

        assert list(pred.shape) == [1, 10], "Shape check"

    @pytest.mark.parametrize(
        "model_input", [torch.rand([1, 28, 28]), torch.rand([1, 28, 28])]
    )
    def test_two(self, model_input):
        model = Net()
        pred, _ = model(model_input)

        assert list(pred.shape) == [1, 10], "Shape check"  # DEBUG
