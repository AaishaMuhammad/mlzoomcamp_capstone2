# Imports

import numpy as np
import onnxruntime as rt
from PIL import Image

# Label and model definitions

labels = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']
onnx_model = "landscape_model_resnet50.onnx"

# Helper Functions

# Loads model session
def model_loader():
    session = rt.InferenceSession(onnx_model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return input_name, output_name, session

# Transforms images for model to process
def image_transformer(path: str, size: int) -> np.ndarray:
    image = Image.open(path)
    image = image.resize((size, size))
    
    image = np.array(image)
    image = image.transpose(2,0,1).astype(np.float32)
    image /= 255

    image = image[None, ...]

    return image

# Final prediction function
def predict(path):
    img = image_transformer(path, 224)
    inputs, outputs, session = model_loader()
    results = session.run([outputs], {inputs: img})[0]
    label = labels[np.argmax(results)]
    return label