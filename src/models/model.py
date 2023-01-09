import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # Architecture: https://nextjournal.com/gkoehler/pytorch-mnist

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        if x.ndim != 4:
            raise ValueError('Expected input to be a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1,28,28]')

        """
        
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))

        x2 = x1.view(-1, 320)
        x2 = F.relu(self.fc1(x2))
        x2 = F.dropout(x2, training=self.training)
        x2 = self.fc2(x2)

        return x2, x1
