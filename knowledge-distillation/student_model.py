# Create student model inheriting from torch.nn.Module

import torch
import torchvision.models as models

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
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
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
    
    def learn(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        b, c, _, _ = y_pred.shape
        y_pred = torch.transpose(y_pred.reshape(b, c, -1), 1, 2)        
        l = self.loss(y_pred, y)
        loss_val = l.item()
        l.backward()
        self.optimizer.step()
        return loss_val
    

class StudentModelPretrained(torch.nn.Module):

    def __init__(self):
        super(StudentModelPretrained, self).__init__()

        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, pretrained=True)
        # for param in model.parameters():
            # param.requires_grad = False

        module_list = list(model.features.children())
        self.mobilenet = torch.nn.Sequential(*module_list[:-4])

        self.conv1 = torch.nn.Conv2d(112, 256, 1, 1, 0)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(256, 256, 1, 1, 0)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(256)
        # reshape operation before the conv operation
        self.conv3 = torch.nn.Conv2d(1, 3, 7, 1, 3)
        self.bn3 = torch.nn.BatchNorm2d(3)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(3, 3, 1, 1, 0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.optimizer.zero_grad()

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        output = self.mobilenet(x)
        output = self.relu1(self.bn1(self.conv1(output)))
        output = self.relu2(self.bn2(self.conv2(output)))
        # reshape the 256 channel vector into 16X16 patch
        # Merge the 16X16 patch into the 23X32 size image. 
        # The resulting image is a 364X512 dimensional, 
        # which is the original size of the input image
        batch_size, channels, height, width = output.size()
        patch_size = 16
        assert channels == patch_size ** 2
        # output = output.view(batch_size, patch_size, patch_size, height, width)
        # output = output.permute(0, 3, 1, 4, 2)
        # output = output.contiguous().view(batch_size, 1, 23 * 16, 32 * 16)
        output = torch.nn.functional.pixel_shuffle(output, patch_size)
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.conv4(output)
        return output

    def learn(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        b, c, _, _ = y_pred.shape
        y_pred = torch.transpose(y_pred.reshape(b, c, -1), 1, 2)        
        l = self.loss(y_pred, y)
        loss_val = l.item()
        l.backward()
        self.optimizer.step()
        return loss_val
