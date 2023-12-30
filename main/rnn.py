import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        self.relu1 = nn.LeakyReLU(0.33)

        self.bn1 = nn.BatchNorm2d(1)

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        class GetRNNOutput(nn.Module):
            def forward(self, x):
                out, _ = x
                return out

        self.recurent_layer1 = nn.Sequential(
            nn.LSTM(input_size=500, hidden_size=25, bidirectional=True, num_layers=2),
            GetRNNOutput(),
        )

        self.recurent_layer2 = nn.Sequential(
            nn.LSTM(input_size=50, hidden_size=25, bidirectional=True, num_layers=2),
            GetRNNOutput(),
        )

        self.ln1 = nn.Sequential(nn.Linear(50, 64), nn.ELU())

        self.bn2 = nn.Sequential(nn.BatchNorm2d(1), nn.Dropout(p=0.5))

        self.ln2 = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 256), nn.Softmax())

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.pool1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)

        # x = x.view(x.size(0), -1)

        # x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc2(x)

        return x
