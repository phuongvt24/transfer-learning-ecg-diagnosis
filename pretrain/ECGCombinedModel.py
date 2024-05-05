import torch
import torch.nn as nn

class ECGCombinedModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ECGCombinedModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(256, 256, batch_first=True, num_layers=3, dropout=0.2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
