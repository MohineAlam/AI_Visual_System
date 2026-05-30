import argparse
from models.model_training import train_model

####==== command line arguments ====####
parser = argparse.ArgumentParser(
  description = "Argumments to provide training and validating data sets, as well as the number of classes used in dataset."
)
parser.add_argument("-t", "--train", required=True, help="Input path to training dataset")
parser.add_argument("-v", "--validate", required=True, help="Input path to validating dataset")
parser.add_argument("-c", "--class_nu", type=int, required=True, help="Number of classes you have in your dataset")
args = parser.parse_args()


####==== check pathways ====####
def check_path(train, validate, class_nu):
  if not os.path.exists(train):
    print(f"Pathway does not exist: {train}")
    sys.exit()
  elif not os.path.exists(validate):
    print(f"Pathway does not exist: {validate}")
    sys.exit()
  elif not isinstance(class_nu, int):
    print(f"The class input is not a number: {class_nu}")
    sys.exit()
  else:
    print("Pathways exist.")
    return True

# call function
if __name__ == "__main__":
    check_path(args.train, args.validate, args.class_nu)
    train_model(args.train, args.validate, args.class_nu)