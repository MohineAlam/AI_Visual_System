# load libraries
from torchvision import datasets, transforms, models # ready to use image data sets, image preprocessing tools, pre-trained deep learnign model
from torch.utils.data import DataLoader # wraps data sets and handles how to train data
import torch.nn as nn # neural network building blocks e.g. loss of fucntion
import torch # core pytorch package e.g. compute gradients
import torch.optim as optim # optimisation algorithm, train nn e.g. adam
import os
import argsparse
import sys

####==== command line arguments ====####
parser = argparse.ArgumentParser(
  description = "Argumments to provide training and validating data sets, as well as the number of classes used in dataset."
)
parser.add_argument("-t", "--train", required=True, help="Input path to training dataset")
parser.add_argument("-v", "--validate", required=True, help="Input path to validating dataset")
parser.add_argument("-c", "--class_nu", type=int, required=True, help="Number of classes you have in your dataset")
args = parser.parse_args()

####==== check pathways ====####
def check_path(train, validate, class_nu):
  if not os.path.exists(train):
    print(f"Pathway does not exist: {train}")
    sys.exit()
  elif not os.path.exists(validate):
    print(f"Pathway does not exist: {validate}")
    sys.exit()
  elif not isinstance(class_nu, int):
    print(f"The class input is not a number: {class_nu}")
    sys.exit()
  else:
    print("Pathways exist.")
    return True

#==========================
def train_model(train, validate, class_nu):
  ####==== transform parameters ====####
  preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
  ])

  ####==== load data set====####
  train_dataset = datasets.ImageFolder(train) # write the path to where you store your training images
  valid_dataset = datasets.ImageFolder(validate) # write the path to where your validating images

  ####==== modify pretrained model ====####
  model = models.resnet18(pretrained=True) # using resnet 18 model  
  model.fc = nn.Linear(model.fc.in_features, class_nu) # give back your classes not pretrained model's 1000

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

# call function
train_model(args.train, args.validate, args.class_nu)
