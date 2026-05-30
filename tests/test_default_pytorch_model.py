
####---- import libraries ----####
from lucam import Lucam
from torchvision import transforms, models
from camera.camera_functions import take_snapshot, process_img
import cv2
import torch
import os
import argparse
import sys
import urllib.request
#================================#

####---- command line arguments ----####
parser = argparse.ArgumentParser(
  description = "Argument for output folder. This is where the images will be saved."
)
parser.add_argument("-o","--output",required=True,help="Output pathway to store images. Use forward slashes.")
args = parser.parse_args()

####---- check if argument exists ----####
def check_path(output):
  if not os.path.exists(output):
    print(f"Your output pathway does not exist: {output}")
    sys.exit()
  else:
    print(f"Your output pathway exists: {output}")
    return True

####---- run model ----####
def run_model(output):
  input_tnsr = preprocess_img(output)
  
  # import pretrained model
  model = models.resnet18(pretrained = True)
  model.eval()
  # make the prediction
  with torch.no_grad(): # simple model run
    output_predict = model(input_tnsr)
    predicted_class = output_predict.argmax().item()

  print(f"Predicted class: {predicted_class}")

  # map ID
  url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
  classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
  print("Predicted label: ", classes[predicted_class])

# call functions
run_model(args.output)
