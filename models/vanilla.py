import torch.nn as nn
import torch.nn.functional as F


class VanillaClassification(nn.Module):
    def __init__(self, num_class: int = 1):
        super().__init__()
        self.model_name = "SimpleClassification"
        self.input_size = 400
        self.model_version = 1
        self.num_class = num_class

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 48 * 48, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, self.num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 48 * 48)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
