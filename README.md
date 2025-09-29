# AI-visual-system
## This repository contains the code for the AI visual system I put together

In this project I used the following 
hardware:
- Lumenera LU200c industrial camera (including USB-B to USB-A cable and charger)
- Raspberry Pi 4 b+ (including micro SD and charger)

And software:
- python v3.10 (download from official website: https://www.python.org/)
-  Software Developer Kit (SDK) v6.9.0 from Lucam Teledyne (link for windows from official website: https://www.teledynevisionsolutions.com/en-gb/support/support-center/software-firmware-downloads/lumenera/lucam-software-and-software-development-kit/)
- lucam (wrapper library)
- opencv-python (image/video processing includnig computer vision library)
- numpy (powers image processing with maths within arrarys e.g. ([1,2,3],[4,5,6]))
- torch and torchvision - from pytorch (deep learning framework and computer vision libraries - image classification and object detection)


### Step One
- Download python v3.10 so it is compatibly with  the lucam software as it uses older library version
- create a virtual environment with the correct python version e.g.:
  python -3.10 -m venv lucam-env
- activate the envrionment
  source lucam-env/Scripts/activate
- Download the SDK from the official website (for your operating system - linux / windows)
- pip install the following libraries:
  pip install opencv-python numpy torch torchvision lucam

