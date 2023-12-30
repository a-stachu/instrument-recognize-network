import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes, case):
        super(Net, self).__init__()
        self.case = case
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)
        # Fully connected layers
        if case == 1000:
            mat2 = 51200
        if case == 100:
            mat2 = 4096
        if case == 10:
            mat2 = 2048

        self.fc1 = nn.Linear(mat2, 11)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(11, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        if self.case == 1000 or self.case == 100:
            x = self.pool2(x)
        else:
            x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
