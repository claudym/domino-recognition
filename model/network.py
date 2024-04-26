import torch
import torch.nn as nn

class DominoCNN(nn.Module):
    def __init__(self):
        super(DominoCNN, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 1000)  # Adjust size according to your input image size
        self.fc2 = nn.Linear(1000, 201)  # Adjusted for 201 classes

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # Flatten the output for the dense layer
        out = self.fc1(out)
        out = self.fc2(out)
        return out
