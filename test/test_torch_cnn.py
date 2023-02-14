import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer3D(nn.Module):
    def __init__(self, num_classes):
        super(Transformer3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)

        self.fc = nn.Linear(128 * (768//8) * (768//8) * (1028//8), num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, 128 * (768//8) * (768//8) * (1028//8))
        x = self.fc(x)

        return x
