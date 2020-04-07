import os
import numpy as np
from PIL import Image

import tensorflow.keras.utils as utils
import torchvision

from onnx_tf.backend import prepare
import onnx

ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_PATH = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.onnx")
DEST = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.pb")


# load model in ONNX format
model_onnx = onnx.load(MODEL_PATH)


# Convert to tensorflow format
prepared_backend = prepare(model_onnx)
prepared_backend.export_graph(DEST)


# test on example image
def loadImage(url):
    filename = url.split('/')[-1]
    img = utils.get_file(filename, url)
    img = Image.open(img)
    return img

img = loadImage("https://ichef.bbci.co.uk/food/ic/food_16x9_832/recipes/fivespicespareribs_70976_16x9.jpg")

class CustomTransform:
    def __init__(self, image_size):
        self.image_size = image_size
        
    def __call__(self, img):
        # resize to a max of IMAGE_SIZE
        w, h = img.size
        scale = min(self.image_size/w, self.image_size/h)
        img = torchvision.transforms.functional.resize(img, (int(h*scale), int(w*scale)))
        
        # add padding to a size of IMAGE_SIZE
        img = torchvision.transforms.functional.pad(img, (0, 0, self.image_size-int(w*scale), self.image_size-int(h*scale)))
        
        # convert to tensor
        img = torchvision.transforms.functional.to_tensor(img)

        # normalize
        img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return img

transform = CustomTransform(image_size=512)

img = transform(img)
images = img.unsqueeze(0).numpy()


# run image on caffe 2 model
x = {model.graph.input[0].name: images}
outputs = prepared_backend.run(x)[0]
print(outputs)