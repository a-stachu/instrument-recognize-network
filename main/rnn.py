import torch.nn as nn


class DefaultRNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(DefaultRNN, self).__init__()

        class GetRNNOutput(nn.Module):
            def forward(self, x):
                out, _ = x
                return out

        self.rnn1 = nn.Sequential(
            nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True),
            GetRNNOutput(),
        )
        self.rnn2 = nn.Sequential(
            nn.RNN(input_size=128, hidden_size=hidden_size, batch_first=True),
            GetRNNOutput(),
        )

        self.ln1 = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.Dropout(0.5)
        )

        self.ln2 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(64), nn.Dropout(0.5))

        self.ln3 = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(64), nn.Dropout(0.5))
        self.fc1 = nn.Linear(16, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(1)
        # print(x.shape)
        x = self.rnn1(x)
        # print(x.shape)
        x = self.rnn2(x)
        # print(x.shape)
        x = self.ln1(x)
        # print(x.shape)
        x = self.ln2(x)
        # print(x.shape)
        x = self.ln3(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = x[:, 0, :]
        # print(x.shape)

        return x


class SimpleRNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(SimpleRNN, self).__init__()

        class GetRNNOutput(nn.Module):
            def forward(self, x):
                out, _ = x
                return out

        self.rnn1 = nn.Sequential(
            nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True),
            GetRNNOutput(),
        )

        self.ln1 = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.ln2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))

        self.fc1 = nn.Linear(64, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.rnn1(x)
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = x[:, 0, :]

        return x
