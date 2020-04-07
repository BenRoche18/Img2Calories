
import os
import onnx
import onnx_tf

ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_PATH = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.onnx")
DEST = os.path.join(ROOT_DIR, "models", "serialized", "mobilenet.pb")


# load model in ONNX format
onnx_model = onnx.load(MODEL_PATH)


# Convert to caffe2 format
prepared_backend = caffe2.python.onnx.backend.prepare(onnx_model)
#prepared_backend.export_graph(DEST)