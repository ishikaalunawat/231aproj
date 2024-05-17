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
        self.fc3 = torch.nn.Conv2d(512, 3, 1, 1, 0)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.optimizer.zero_grad()

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def learn(self, x, y, batch_size=8, epochs=100):
        assert x.shape[0] == y.shape[0]
        for _ in epochs:
            y_pred = self.forward(x[: batch_size])
            b, c, _, _ = y_pred.shape
            y_pred = torch.transpose(y_pred.reshape(b, c, -1), 1, 2)
            self.zero_grad()
            self.loss(y_pred, y[: batch_size]).backward(retain_graph=True)
            self.optimizer.step()
