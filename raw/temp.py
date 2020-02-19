import os
import numpy as np
import onnx

ROOT_DIR = os.path.dirname(os.getcwd())

MODEL_PATH = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.onnx")

CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food256", "condensed-category.txt")


# extract class names
CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food256", "condensed-category.txt")
with open(CLASS_PATH, 'r') as file:
	file.readline()
	class_names = [line.split('\t')[1].strip() for line in file.readlines()]


model = onnx.load(MODEL_PATH)
print(onnx.checker.check_model(model))