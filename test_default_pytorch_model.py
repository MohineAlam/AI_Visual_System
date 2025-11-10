
####---- import libraries ----####
from lucam import Lucam
from torchvision import transforms, models
import cv2
import torch
import os
import argparse
import sys
import urllib.request
#================================#

####---- command line arguments ----####
parser = argparseArgumentParser(
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

####---- take image ----####
def take_snapshot(output):
    if check_path(output):
      cam = Lucam()
      image = cam.TakeSnapshot() # raw grey scale image (single colour per pixel)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB) # demosaic - adds colour
      cv2.imwrite(os.path.join(output,"classed_image.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)) # save picture
      cam.CameraClose()
      return image_rgb
    else:
      print("There was an issue with opening the camera and taking a snapshot.")
      sys.exit()

####---- preprocess image dimention for model ----####
def preprocess_img(output):
  img_rgb = take_snapshot(output)
  
  # preprocessing parameters
  preprocess = transforms.Compose[(
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # converts to torch.Tensor and scale to [0,1]
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
  )]

  # apply changes and add batch size value
  input_tensor = preprocess(img_rgb).unsqueeze(0) # tensor array changed and will be in format [batch_size, channel, height, width]
  return input_tensor

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
