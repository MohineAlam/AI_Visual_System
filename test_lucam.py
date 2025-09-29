# import lucam library
from lucam import Lucam # class we will use
import cv2

# you can check if the class exists by uncommenting
#print(dir(Lucam))

# you can check if the SDK is correct for your OS by uncommecting and inputting your path to SDK
#ctypes.WinDLL(r"\path\to\the\SDK\you\installed\lucamapi.dll")

# initialise the camera
cam = Lucam()
print("Camera initialised: ", cam.name)

# take image and convert to colour scale
image = cam.TakeSnapshot()
rgb_image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
# print shape as numpy array
# safe test using exception handling
try:
  print("Captured image shape: ", image.shape)
except Exception as e:
  print("Error during snapshot: ", e)
  # check what array it is if it is not numpy array
  print("Image type", type(image))

# save image
try:
  cv2.imwrite("test_output.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
  print("Picture saved")
except Exception as e:
  print("Error, image not saved.")

# close the camera
del cam
print("Camera released")
