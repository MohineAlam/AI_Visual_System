# load libraries
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import os

####==== transform parameters ====####
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
])

####==== load data set====####
train_dataset = datasets.ImageFolder("your/path/to/train/folder") # write the path to where you store your training images
valid_dataset = datasets.ImageFolder("your/path/to/train/folder") # write the path to where your validating images

####==== modify pretrained model ====####
model = models.resnet18(pretrained=True) # using resnet 18 model
num_classes = "" # change to number of classes you are using e.g. 1,2, or 3 etc.
model.fc = nn.Linear(model.fc.in_features, num_classes) # give back your classes not pretrained model's 1000

####==== train the model ====####
# change to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss() # calculates how much the prediction is wrong
optimiser = optim.Adam(model.parameters(), lr=le-4) # use optimisation algorithm and adjusts learning rate

# number of iterations to train
for epoch in range(10):
  model.train()
  running_loss = 0.0
  for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimser.zero_grad() # rezero to prevent accumulation of gradients of optimised parameters (direction of corrections)
      outputs = model(images) # model processes the input and produces predictions
      loss = criterion(outputs, labels) # computes error (loss) between model's predictions (output) and labels
      loss.backward() # computes gradients of the loss with model's parameters
      optimiser.step() # updates model's parmeters using the gradients computed in previous step
      running_loss += loss.item() # accumulates loss of current batch for tracking during training
  # print training process
  print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

####==== save trained model ====####
# create folder to save in your directory
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
