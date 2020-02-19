import torch
import torchvision

import os

ROOT_DIR = os.path.dirname(os.getcwd())

MODEL_PATH = os.path.join(ROOT_DIR, "models", "food_classification", "fine_tuning_mobilenet.pt")
DEST = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet_cnn.pt")

# define model architecture
model = torchvision.models.mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 101)


# load model weights
model.load_state_dict(torch.load(MODEL_PATH))


# serialize
model.eval()
example = torch.rand(1, 3, 512, 512)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(DEST)