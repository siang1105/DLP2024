# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()
    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(Nu)
        self.square1 = SquareLayer()
        self.drop1 = nn.Dropout(dropoutRate)

        # Second convolution block
        self.conv2 = nn.Conv2d(1, 20, (Nu, 12), padding=(0, 6))
        self.bn2 = nn.BatchNorm2d(20)
        self.square2 = SquareLayer()
        self.drop2 = nn.Dropout(dropoutRate)

        # Pooling block
        self.pool = nn.AvgPool2d((1, 62), stride=(1, 12))

        # Get size of the fully connected layer input
        fc_input_size = self.get_size(C, timeSample)
        
        # Define fully connected layer with calculated input size
        # self.fc = nn.Linear(640, numClasses)
        self.fc = nn.Linear(fc_input_size, numClasses)

    # Forward Pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.square1(x)
        x = self.drop1(x)

        x = x.permute(0, 2, 1, 3)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.square2(x)
        x = self.drop2(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return F.log_softmax(x, dim=1)

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        with torch.no_grad():
            self.fc = nn.Identity()
            fc_input_size = self.forward(torch.zeros(1, 1, C, N)).shape[1]
            self.fc = None  # Reset self.fc
            return fc_input_size


if __name__ == '__main__':
    model = SCCNet()
    print(model)