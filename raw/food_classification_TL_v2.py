import torch
import torchvision
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import shutil


# ENABLE GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


# TRAINING PARAMETERS
IMAGE_SIZE = 299
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


# FETCH FOOD-101 DATASET
FOOD101_DIR = os.path.join("data", "food101")
TRAIN_DIR = os.path.join(FOOD101_DIR, "train")
VAL_DIR = os.path.join(FOOD101_DIR, "test")


# FORMAT TRAINING IMAGES
train_transform = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.CenterCrop(IMAGE_SIZE),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_raw = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
train_size = len(train_raw)

train_loader = torch.utils.data.DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True)

class_names = train_raw.classes


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


# SHOW EXAMPLE IMAGES
cols, rows = 5, 4
fig = plt.figure(figsize=(20,10))
fig.suptitle("Random images from dataset")

for i in range(1, cols*rows+1):
  fig.add_subplot(rows, cols, i)

  # randomly sellect image from dataset
  j = np.random.randint(train_size)
  img = train_raw[j][0].numpy()
  img = np.transpose(img, (1,2,0))
  img = img * (0.485, 0.456, 0.406) + (0.229, 0.224, 0.225)

  plt.title(class_names[train_raw[j][1]])
  plt.axis("off")
  img = np.clip(img, 0, 1)
  plt.imshow(img, interpolation="nearest")
plt.show()


# FETCH MODEL
print("Fetching model...", end="")
model = torchvision.models.inception_v3(pretrained=True)

# freeze feature extraction layers
for param in model.parameters():
  param.requires_grad = False

# handle auxilary net
num_features = model.AuxLogits.fc.in_features
model.AuxLogits.fc = torch.nn.Linear(num_features, 101)

# handle primary net
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 101)

model = model.to(device)

print("done")


# DECLARE OPTIMIZER AND LOSS FUNCTION
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.CrossEntropyLoss()


# TRAIN VIA TRANSFER LEARNING
statistics = {
    "accuracy": [],
    "val_accuracy": []
}

def train():
  best_acc = 0.0
  best_model_weights = None

  print("Training on {} images...\n".format(train_size))

  for epoch in range(1, EPOCHS+1):
    print("Epoch {}/{}...".format(epoch, EPOCHS))

    ### TRAINING PHASE ###
    model.train()
    running_loss = 0.0
    running_corrects = 0

    i = 0
    for (images, labels) in train_loader:
      images = images.to(device)
      labels = labels.to(device)

      # reset the parameter gradients
      optimizer.zero_grad()

      # fit images on model
      outputs, aux_outputs = model(images)
      _, predictions = torch.max(outputs, 1)
      loss1 = loss_func(outputs, labels)
      loss2 = loss_func(aux_outputs, labels)
      loss = loss1 + 0.4 * loss2

      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      running_corrects += torch.sum(predictions == labels.data)

      i += 1
      print("Training Loss: {:.4f}, Acc: {:.4f}".format(running_loss / (i * BATCH_SIZE), running_corrects.double() / (i * BATCH_SIZE)))

    # calculate statistics
    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    
    # print statistics
    print("Training Loss: {:.4f}, Acc: {:.4f}".format(epoch_loss, epoch_acc))
    statistics['accuracy'].append(epoch_acc)

    val_loss, val_acc = validate()

    # save best model seen
    if val_acc > best_acc:
      best_acc = epoch_acc
      best_model_weights = copy.deepcopy(model.state_dict())

    # print statistics
    print("Validation Loss: {:.4f}, Acc: {:.4f}\n".format(val_loss, val_acc))
    statistics['val_accuracy'].append(val_acc)

  # reinstantiate best seen weights
  model.load_state_dict(best_model_weights)

  print("DONE")

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

train()


# SAVE MODEL WEIGHTS
MODEL_PATH = os.path.join("models", "food_classifier_TL_v2.pt")
torch.save(model.state_dict(), MODEL_PATH)


# EVALUATE
plt.plot(statistics['accuracy'], label='accuracy')
plt.plot(statistics['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid(True)
plt.show()