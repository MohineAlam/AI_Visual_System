from lucam import Lucam
from torchvision import transforms, models


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
  preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # converts to torch.Tensor and scale to [0,1]
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
  ])

  # apply changes and add batch size value
  input_tensor = preprocess(img_rgb).unsqueeze(0) # tensor array changed and will be in format [batch_size, channel, height, width]
  return input_tensor