import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, case):
        super(SimpleCNN, self).__init__()
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

        self.fc1 = nn.Linear(mat2, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # print(x.shape)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)

        if self.case == 1000 or self.case == 100:
            x = self.pool2(x)
        else:
            x = self.pool3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # print(x.shape)
        return x


class DefaultCNN(nn.Module):
    def __init__(self, num_classes, case):
        super(DefaultCNN, self).__init__()
        self.case = case
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.dp1 = nn.Dropout(0.3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
        )

        if case == 1000:
            mat2 = 102400
        if case == 100:
            mat2 = 8192
        if case == 10:
            mat2 = 4096

        self.ln1 = nn.Sequential(
            nn.Linear(mat2, 128), nn.BatchNorm1d(128), nn.Dropout(0.5)
        )

        self.ln2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(0.5)
        )

        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        if self.case == 1000 or self.case == 100:
            x = self.pool1(x)
        else:
            x = self.pool2(x)
        x = self.dp1(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        # print(x.shape)

        x = self.ln1(x)
        # print(x.shape)
        x = self.ln2(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)

        return x
