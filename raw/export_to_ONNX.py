import tensorflow


# export model to ONNX
torch.onnx.export(model, images, DEST, opset_version=11, keep_initializers_as_inputs=True)