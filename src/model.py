import torch.nn as nn
import torch.nn.functional as F

class PlantCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.fc1 = nn.Linear(32 * 60 * 60, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x