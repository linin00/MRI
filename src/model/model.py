import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    # 1*128*128*128
    self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
    self.pool1 = nn.MaxPool3d(2, 2, ceil_mode=True)
    self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
    self.pool2 = nn.MaxPool3d(2, 2, ceil_mode=True)
    self.fc1 = nn.Linear(32*8*8*8, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))    # input(1, 128, 128,128) output(16, 65, 65,65)
    print(x.shape)
    x = self.pool1(x)            # output(16, 32, 32,32)
    print(x.shape)
    x = F.relu(self.conv2(x))    # output(32, 16, 16)
    x = self.pool2(x)            # output(32, 8, 8)
    print(x.shape)
    x = x.view(x.size(0), -1)    # output(32*8*8*8)
    print(x.shape)
    x = F.relu(self.fc1(x))      # output(120)
    x = F.relu(self.fc2(x))      # output(84)
    x = self.fc3(x)              # output(10)
    return x
