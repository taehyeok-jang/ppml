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

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Router:
  def __init__(self):
    self.count = 0

  def validate(self, model_name: str):
    if model_name in [
       # torchvision
       'resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'densenet121',
       'densenet201', 'mobilenet_v2', 'inception_v3', 'efficientnet_b0',
       'efficientnet_b7', 'squeezenet1_0', 'alexnet', 'googlenet', 'shufflenet_v2_x1_0'
       # timm 
       'vit_base_patch16_224', 'vit_large_patch16_224', 'deit_base_patch16_224',
       'convnext_base', 'convnext_large'
       ]:
        pass
    else:
        raise ValueError(f"Model {model_name} not available.")

  def route_request(self, model_name: str, image_payload_bytes: bytes):
    """
    Route the classification request to the appropriate model deployment.
    """
    model_endpoint = f"http://localhost:8000/{model_name}/classify_"
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
