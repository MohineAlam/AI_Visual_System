# load library
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from lucan import Lucam
import argparse
import os
import sys
import cv2
import torch.nn as nn
import torch

####---- trace functions being called ----####
def trace_my_code(frame,event,arg):
  """
  Traces the functions being called
  Returns trace to track all functions
  """
  if os.path.basename(frame.f_code.co_filename) == "ML_Image_Decipherer.py":
    if event == "call":
      print(f"-> Calling {frame.f_code.co_name}() at line {frame.f_lineno}")
    elif event == "return":
      print(f"<- Returning {frame.f_code.co_name}()")
    
    return trace_my_code
# turn tracing on
sys.settrace(trace_my_code)

####---- command line arguments ----####
parser = argparse.ArgumentParser(
  description = "Arguments required to call functions." 
)
parser.add_argument("-i", "--input", required = True, help = "Name the input image will be saved under.")
parser.add_argument("-o", "--output", required = True, help = "Output pathway to store your input image.")
parser.add_argument("-m", "--model_path", required = True, help = "Pathway where your trained model is stored. Inlucde the name of the model, e.g., model.pth")
parser.add_argument("-c", "--class_names", required = True, help = "This is a list of the class names you initially used to train your model e.g. ['Dog', 'Cat', 'Puppy', 'Kitten']. The names must be the same.")

args = parser.parse_args()

####---- check pathway ----####
def check_path(output, model_path):
  """ 
  Function to check paths 
  parameters: output, model_path
  returns True
  """
  if not os.path.exists(output, model_path):
    print("Output path does not exist.")
    sys.exit()
  elif not os.path.exists(model_path):
    print("Pathway to model does not exist.")
    sys.exit()
  else:
    print("Output and model pathway exists.")
    
    return True

####---- take snapshot ----####
def take_snapshot(output,input):
  """ 
  Function to take image and save it
  parameters: output, input
  returns grey scale and coloured image
  """
  if check_path(output):
    cam = Lucam()
    print("Camera initialised: ", cam.name)
    cam.PixelFormat = Lucam.PIXEL_FORMAT["8"]
    image = cam.TakeSnapshot()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
  else:
    logging.error("Error taking snapshot.")
    print("Check Camera set up.")
    sys.exit()
  # save image
  try:
    name = (input, ".jpg")
    save_img = (os.path.join(output, "".join(name)))
    cv2.imwrite(save_img, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(f"Image saved in: {save_img}")
    cam.CameraClose()
  except Exception as e:
    logging.error("Error during snapshot: {e}")
    sys.exit()
  
  return image, rgb_image

####---- preprocess image for model - formatting ----####
def preprocess_img(output,input):
  """
  Transforms image to suit pytorch libraries
  parameters: output, input
  returns modified 4d tensor
  """
  # unpack snapshot 
  img, rgb_img = take_snapshot(output,input)
  # formatting parameters
  preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
  ])
  # apply changed
  input_tensor = preprocess(rgb_img).unsqueeze(0)
  
  return input_tensor

####---- run trained model ----####
def run_model(output, input, class_names, model_path):
  """
  This function runs the model after calling the previous functions
  unpacks tensor to predict with model
  parameters: output, input, class_name, model_path
  returns predicted class and label
  """
  # unpack tensor
  tensor = preprocess_img(output, input)
  # make prediction
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # change to gpu if available
  model = models.resnet18(pretrained = False)
  model.fc = nn.Linear(model.fc.in_features, len(class_names))
  load_model = torch.load(model_path)
  model.load_state_dict(load_model)
  model.to(device)
  model.eval()

  with torch.no_grad(): # don't train model with image
    tensor = tensor.to(device) # put tensor on same computer (GPU or CPU)
    output = model(tensor)
    predicted_class = output.argmax().items()
    predicted_label = class_names[predicted_class]
  # print prediction
  print(f"Predicted class: {predicted_class}")
  print(f"Predicted label: {predicted_label}")

####---- call function ----####
user = input("All set up and ready to take a snapshot with your camera? Press y to continue or anything else to exit: ")
if user == "y":
  print("Running model.")
  run_model(args.output, args.input, args.class_names, args.model_path)
else:
  print("Exiting system.")
  sys.exit()
