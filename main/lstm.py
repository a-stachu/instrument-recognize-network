import torch.nn as nn


class DefaultLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(DefaultLSTM, self).__init__()

        class GetRNNOutput(nn.Module):
            def forward(self, x):
                out, _ = x
                return out

        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True),
            GetRNNOutput(),
        )

        self.lstm2 = nn.Sequential(
            nn.LSTM(input_size=256, hidden_size=hidden_size, bidirectional=True),
            GetRNNOutput(),
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(0.5)

        self.ln1 = nn.Sequential(nn.Linear(256, 128), nn.ELU())

        self.ln2 = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(128, 64), nn.ELU())

        self.ln3 = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 32), nn.Softmax())

        self.fc1 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.fc1(x)
        x = x[:, 0, :]

        return x
