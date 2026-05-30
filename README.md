# AI-Visual-System
## In this repository you will find the pipeline for setting up a machine learning visual system - this will predict what the image is!

## In this project I use the following: 
### Hardware
- Lumenera LU200c industrial camera (including USB-B to USB-A cable and charger)

### Software
#### Camera driver
-  Software Developer Kit (SDK) v6.9.0 from Lucam Teledyne (link for windows from official website: https://www.teledynevisionsolutions.com/en-gb/support/support-center/software-firmware-downloads/lumenera/lucam-software-and-software-development-kit/)
#### python libraries
Main libraries for interfacing with the drive and setting up the machine learning model (install using conda and the enviornment.yml file):
- lucam (wrapper library which interacts with the lumenera LuCam API)
- opencv-python (image/video processing includnig computer vision library)
- torch and torchvision - from pytorch (deep learning framework and computer vision libraries - image classification and object detection)

### Step One
- Clone this github respository using the clone link etc.
  
- Set up and activate your environemnt, and install required packages using the environment.yml:
``` bash
  conda env create -f environment.yml 
```
- Download the SDK from the official website (for your operating system - linux / windows)

### Step Two
- set up your lumenera camera with the USB-B to USB-A cable connecting it to your PC (plug in the barrel jack charger if required - usually the USB cable is enough to power it)
- Run the test_lucam.py script, this will test if the wrapper library works with the SDK:
  python test_lucam.py

If successful you should recieve a message in the follownig structure:
"Camera initialised: pUSHORT"
"Caputured image shape: (1200, 1600)"
"Image saved"
"Camera released"

If unsuccessful:
"Error during snapshot: (error type)"
"Image type: (image type)"
"Error, image not saved"

### Step Three
- Place the camera infront of a clear object
- run the test_default_pytorch_model.py script to see if the resnet and pytorch has been set up correctly
  python test_default_pytorch_model.py -o /path/to/output/folder

If successful you should recieve the predicted class and label in this format:
"Predicted class: "
"Predicted label: "

### Step Four
- train your modified resnet18 model using pytorch, you will provide the paths to your training and validating datasets, as well as the number of classes you have within
- run the train_your_model.py script:
  python train_your_model.py -t /path/to/training/dataset -v /path/to/validate/dataset -c 5

If successful you will have your model saved in a new folder called "models/" within your working directory called model.pth

### Step Five
- Run your trained model on a live snapshot, set up your camera and run the ML_image_decipherer.py script with four arguments:
  python ML_image_decipherer.py -i image_name -o output/path/to/save/image -c ["list","of","class","names"] -m model/path/model.pth
- There will be a prompt asking if you're "All set up and ready to take a snapshot with your camera?", you can then press y to continue.

If successful you will have a prediction in the format:
  "Predicted class: "
  "Predicted label: "
  
