# Create student model inheriting from torch.nn.Module

import torch

class StudentModel(torch.nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Define a Fully Convolutional Network layer
        self.conv1 = torch.nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, 1, 2)
        self.conv3 = torch.nn.Conv2d(128, 256, 5, 1, 2)
        self.conv4 = torch.nn.Conv2d(256, 512, 5, 1, 2)
        self.conv5 = torch.nn.Conv2d(512, 512, 1, 1, 0)
        self.conv6 = torch.nn.Conv2d(512, 512, 5, 1, 2)

        self.fc1 = torch.nn.Conv2d(512, 512, 1, 1, 0)
        self.fc2 = torch.nn.Conv2d(512, 512, 1, 1, 0)
        self.fc2 = torch.nn.Conv2d(512, 3, 1, 1, 0)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.optimizer.zero_grad()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def learn(self, x, y):
        y_pred = self.forward(x)
        loss = torch.nn.MSELoss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()