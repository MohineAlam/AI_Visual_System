# import lucam library
from lucam import Lucam # class we will use

# you can check if the class exists by uncommenting
#print(dir(Lucam))

# you can check if the SDK is correct for your OS by uncommecting and inputting your path to SDK
#ctypes.WinDLL(r"\path\to\the\SDK\you\installed\lucamapi.dll")

# initialise the camera
cam = Lucam()
print("Camera initialised: ", cam.name)

# take image
image = cam.TakeSnapshot()

# print shape as numpy array
# safe test using exception handling
try:
  print("Captured image shape: ", image.shape)
except:
  print("Error during snapshot: ", e)
  # check what array it is if it is not numpy array
  print("Image type", type(image))

# save image
cam.SaveImage(image, "test_output.bmp")

# close the camera
del cam
print("Camera released")
