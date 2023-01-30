'''
citations:
1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data
2. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  data_set=datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
  loader = torch.utils.data.DataLoader(data_set, batch_size = 64, shuffle=training)
  return loader

def build_model():
  model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(784, 128), #28*28 = 784
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
  )

  return model

def train_model(model, train_loader, criterion, T):
  model.train()
  opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
  for epoch in range(T):
    agg_loss = 0.0
    accurate = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
      input, label = data
      opt.zero_grad()

      output = model(input)

      _, predicted = torch.max(output.data, 1)
      total += label.size(0)
      accurate += (predicted == label).sum().item()

      loss = criterion(output, label)
      loss.backward()
      opt.step()

      agg_loss += loss.item() * train_loader.batch_size 
    
    print(f'Train Epoch: {epoch}\t Accuracy: {accurate}/{total}({(100 * accurate / total):.2f}%)\t Loss: {(agg_loss/len(train_loader.dataset)):.4f}')

def evaluate_model(model, test_loader, criterion, show_loss = True):
  model.eval()
  agg_loss = 0.0
  accurate = 0.0
  total = 0.0
  with torch.no_grad():
    for input, label in test_loader:
      output = model(input)

      _, predicted = torch.max(output.data, 1)
      total += label.size(0)
      accurate += (predicted == label).sum().item()
      loss = criterion(output, label)
      agg_loss += loss.item() * test_loader.batch_size
    if(show_loss): 
      print(f'Average loss: {(agg_loss/len(test_loader.dataset)):.4f}')
    print(f'Accuracy: {(100 * accurate / total):.2f}%')


def predict_label(model, test_images, index):
  class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
  logits = model(test_images[index])
  probabilities = F.softmax(logits, dim=1)
  prob_list = probabilities.tolist()
  flatten_prob_list = [item for sublist in prob_list for item in sublist]
  map = {class_names[i] : flatten_prob_list[i]*100 for i in range(len(class_names))}
  sorted_map = sorted(map.items(), key=lambda x:x[1], reverse=True)
  for i in range(3):
    print(f'{sorted_map[i][0]}: {sorted_map[i][1]:.2f}%')

if __name__ == '__main__':

  print('-------------part 1: get_data_loader---------------')
  train_loader = get_data_loader()
  print(type(train_loader))
  print(train_loader.dataset)

  print('-------------part 2: build_model---------------')
  model = build_model()
  print(model)

  print('-------------part 3: train_model---------------')
  criterion = nn.CrossEntropyLoss()
  train_model(model, train_loader, criterion, T = 5)

  print('-------------part 4: evaluate_model---------------')
  test_loader = get_data_loader(training = False)
  evaluate_model(model, test_loader, criterion, show_loss = True)

  print('-------------part 5: predict_label---------------')
  images, lables = iter(test_loader).next()
  predict_label(model, images, 6)

  import matplotlib.pyplot as plt
  plt.imshow(images[6].squeeze(), cmap="gray")
  plt.show()
  print(f'actual answer : {class_names[lables[6]]}')
