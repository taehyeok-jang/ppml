import ray
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, UploadFile, File

from PIL import Image

from typing import Dict
from io import BytesIO

import requests

'''
# script for testing 

curl -X POST "http://127.0.0.1:8000/classify_" \
-H "Content-Type: multipart/form-data" \
-F "file=@./grey-British-Shorthair-compressed.jpg"
'''

# https://github.com/chenyaofo/pytorch-cifar-models
SUPPORTED_MODELS = [
 'mobilenetv2_x0_5',
 'mobilenetv2_x0_75',
 'mobilenetv2_x1_0',
 'mobilenetv2_x1_4',
 'repvgg_a0',
 'repvgg_a1',
 'repvgg_a2',
 'resnet20',
 'resnet32',
 'resnet44',
 'resnet56',
 'shufflenetv2_x0_5',
 'shufflenetv2_x1_0',
 'shufflenetv2_x1_5',
 'shufflenetv2_x2_0',
 'vgg11_bn',
 'vgg13_bn',
 'vgg16_bn',
 'vgg19_bn',
# VISION_TRANSFORMERS
 'vit_small_patch16_384',
 'vit_base_patch16_384',
 'vit_large_patch16_384',
# CONVNEXTS
 'convnext-tiny',
 'convnext-base',
 ]

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Router:
  def __init__(self):
    self.count = 0

  def validate(self, model_name: str):
    if model_name in SUPPORTED_MODELS:
        pass
    else:
        raise ValueError(f"Model {model_name} not available.")

  def route_request(self, model_name: str, image_payload_bytes: bytes):
    """
    Route the classification request to the appropriate model deployment.
    """
    model_endpoint = f"http://localhost:8000/v2/{model_name}/classify_"
    try:
        files = {"file": ("image.jpg", BytesIO(image_payload_bytes), "image/jpeg")}
        response = requests.post(model_endpoint, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to query model {model_name}. HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
    
  @app.get("/")
  def get(self):
    return f"Welcome to the model zoo serving system."

  @app.post("/classify_")
  async def classify_(self, model_name:str, file: UploadFile = File(...)):
    self.validate(model_name)
    
    image_bytes = await file.read()
    return self.route_request(model_name, image_bytes)

def builder(args: Dict[str, str]) -> Application:
    """
    Build and return the Ray Serve Application based on arguments from `config.yaml`.
    """
    print(f"Building deployment for model zoo")

    return Router.bind()
