import os
import torch
import torchvision

ROOT_DIR = os.path.dirname(os.getcwd())

MODEL_PATH = os.path.join(ROOT_DIR, "models", "food_detection", "condensed", "mobilenet_backbone.pt")
DEST = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.onnx")

CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food256", "condensed-category.txt")


# extract class names
CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food256", "condensed-category.txt")
with open(CLASS_PATH, 'r') as file:
	file.readline()
	class_names = [line.split('\t')[1].strip() for line in file.readlines()]


# define model architecture
backbone  = torchvision.models.mobilenet_v2(pretrained=False).features
backbone.out_channels = 1280
anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
model = torchvision.models.detection.FasterRCNN(backbone, num_classes=len(class_names), rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


# load model weights
model.load_state_dict(torch.load(MODEL_PATH))


# input random image to model
model.eval()
x = torch.rand(1, 3, 512, 512)
outputs = model(x)


# export model to ONNX
torch.onnx.export(model, x, DEST, opset_version=11)