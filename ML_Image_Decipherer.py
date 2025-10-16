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
args = parser.parse_args()

####---- check pathway ----####
def check_path(output):
  if not os.path.exists(output):
    print("Output path does not exist.")
    sys.exit()
  else:
    print("Output pathway exists.")
    
    return True

####---- take snapshot ----####
def take_snapshot(output,input):
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
