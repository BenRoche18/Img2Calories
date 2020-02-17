import torch
import torchvision
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import shutil

ROOT_DIR = os.path.dirname(os.getcwd())

# ENABLE GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


# VALIDATION PARAMETERS
IMAGE_SIZE = 299
BATCH_SIZE = 32


# FETCH FOOD-101 DATASET
FOOD101_DIR = os.path.join(os.path.abspath(os.sep), "Datasets", "food101")
VAL_DIR = os.path.join(FOOD101_DIR, "test")


# FORMAT VALIDATION IMAGES
val_transform = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.CenterCrop(IMAGE_SIZE),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_raw = torchvision.datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_size = len(val_raw)

val_loader = torch.utils.data.DataLoader(val_raw, batch_size=BATCH_SIZE)

class_names = val_raw.classes


# FETCH MODEL
print("Fetching model...", end="")
model = torchvision.models.inception_v3(pretrained=True)

# handle auxilary net
num_features = model.AuxLogits.fc.in_features
model.AuxLogits.fc = torch.nn.Linear(num_features, 101)

# handle primary net
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 101)

model = model.to(device)
print("done")


# LOAD MODEL WEIGHTS
print("Loading transfer learning weights...", end="")
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", "food_classification", "fine_tuning.pt")))
print("done")


# DECLARE LOSS FUNCTION
loss_func = torch.nn.CrossEntropyLoss()


# VALIDATE TEST IMAGES

def validate():
  model.eval()
  running_loss = 0.0
  running_corrects = 0

  for (images, labels) in val_loader:
    images = images.to(device)
    labels = labels.to(device)

    # fit images on model
    with torch.no_grad():
      outputs = model(images)
      _, predictions = torch.max(outputs, 1)
      loss = loss_func(outputs, labels)

    running_loss += loss.item() * images.size(0)
    running_corrects += torch.sum(predictions == labels.data)

  # calculate statistics
  val_loss = running_loss / val_size
  val_acc = running_corrects.double() / val_size

  return val_loss, val_acc


val_loss, val_acc = validate()
print("Loss: {:.4f}\nAcc: {:.4f}".format(val_loss, val_acc))