import sys
sys.path.append('src/preprocess')
sys.path.append('src/model')
from model import LeNet
from data_transform import Dataset
from torch.autograd import Variable
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np

net = LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

dataset_path = "data/dataset"
dataset = Dataset()
dataset.load_dataset(dataset_path)
imgs, labels = dataset.get_split_dataset()

# 将数据转化为tensor格式，便于输入网络
imgs_f, labels_f = torch.tensor(np.array(imgs)).float(), torch.tensor(np.array(labels)).float()

# Function to save the model
def saveModel():
  path = "data/model.pth"
  torch.save(net.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test imgs
def testAccuracy():
  net.eval()
  accuracy = 0.0
  total = 0.0
  with torch.no_grad():
    outputs = net(imgs_f)
    # the label with the highest energy will be our prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels_f.size(0)
    accuracy += (predicted == labels_f).sum().item()

  # compute the accuracy over all test imgs
  accuracy = (100 * accuracy / total)
  return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
  best_accuracy = 0.0
  # Define your execution device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("The model will be running on", device, "device")
  # Convert model parameters and buffers to CPU or Cuda
  net.to(device)
  for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    imgs = Variable(imgs_f.to(device))
    labels = Variable(labels_f.to(device))
    # zero the parameter gradients
    optimizer.zero_grad()
    # predict classes using imgs from the training set
    outputs = net(imgs)
    # compute the loss based on model output and real labels
    print(outputs.shape, labels.shape)

    loss = loss_fn(outputs, labels)
    # backpropagate the loss
    loss.backward()
    # adjust parameters based on the calculated gradients
    optimizer.step()

    # Let's print statistics for every 1,000 imgs
    running_loss += loss.item()     # extract the loss value 
    # print every 1000 (twice per epoch) 
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, 1, running_loss / 1000))
    # zero the loss
    running_loss = 0.0

    # Compute and print the average accuracy fo this epoch when tested over all 10000 test imgs
    accuracy = testAccuracy()
    print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
    # we want to save the model if the accuracy is the best
    if accuracy > best_accuracy:
        saveModel()
        best_accuracy = accuracy

if __name__ == '__main__':
  train(2)