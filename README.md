# AI-Visual-System
## In this repository you will find the pipeline for the AI visual system 

## In this project I used the following: 
### Hardware
- Lumenera LU200c industrial camera (including USB-B to USB-A cable and charger)
- Raspberry Pi 4 b+ (including micro SD and charger)

### Software
- python v3.10 (download from official website: https://www.python.org/)
-  Software Developer Kit (SDK) v6.9.0 from Lucam Teledyne (link for windows from official website: https://www.teledynevisionsolutions.com/en-gb/support/support-center/software-firmware-downloads/lumenera/lucam-software-and-software-development-kit/)
- lucam (wrapper library which interacts with the lumenera LuCam API)
- opencv-python (image/video processing includnig computer vision library)
- torch and torchvision - from pytorch (deep learning framework and computer vision libraries - image classification and object detection)
- json (to encode and decode JSON data, ease data exchange between systems and API)
- argparse (creates user friendly command line interfaces)


### Step One
- download python v3.10 so it is compatible with the lucam software as it uses older python versions
- create a virtual environment with the correct python version e.g.:
  python -3.10 -m venv lucam-env
- activate the envrionment
  source lucam-env/Scripts/activate
- Download the SDK from the official website (for your operating system - linux / windows)
- pip install the following libraries:
  pip install opencv-python torch torchvision lucam argparse

### Step Two
- set up your lumenera camera with the USB-B to USB-A cable connecting it to your PC (plug in the barrel jack charger if required - usually the USB cable is enough to power it)
- Clone this github respository to access the scripts, using the clone link etc.
- Run the test_lucam.py script, this will test if the wrapper library works with the SDK:
  python test_lucam.py

If successfull you should recieve a message in the follownig structure:
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

If successfull you should recieve the predicted class and labelin this format:
"Predicted class: "
"Predicted label: "

### Step Four
- train your modified resnet18 model using pytorch, you will provide the paths to your training and validating datasets, as well as the number of classes you have within
- run the train_your_model.py script:
- python train_your_model.py -t /path/to/training/dataset -v /path/to/validate/dataset -c 5

If successfull you will have your model saved in a new folder called "models/" within your working directory called model.pth

### Step Five
- Run your trained model on a live snapshot
