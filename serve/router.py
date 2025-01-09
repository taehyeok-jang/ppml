import ray
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, UploadFile, File
import numpy as np 

from PIL import Image

from typing import Dict
from io import BytesIO

import requests

from model_profiles import (
    CIFAR10_model_zoo_profile,
    CIFAR100_model_zoo_profile,
    PARETO_FRONT_MODELS,
    CIFAR10_PARETO_FRONT_SPEC,
    CIFAR100_PARETO_FRONT_SPEC
)

from psml_defense import PsmlDefenseProxy


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
  def __init__(self, dataset: str, eps: float):
    self.count = 0

    pareto_front_models = PARETO_FRONT_MODELS
    if dataset == "cifar10":
        pareto_front_spec = CIFAR10_PARETO_FRONT_SPEC
    elif dataset == "cifar100":
        pareto_front_spec = CIFAR100_PARETO_FRONT_SPEC
    else: 
        raise ValueError(f"Unsupported dataset: {dataset}")

    self.proxy = PsmlDefenseProxy(pareto_front_models=pareto_front_models, pareto_front_spec=pareto_front_spec, eps=eps, sensitivity=1)
    self.utility = 0 # measurement for system utility; in a trade-off relationship between the level of defense

  def validate(self, model_name: str):
    if model_name in SUPPORTED_MODELS:
        pass
    else:
        raise ValueError(f"Model {model_name} not available.")

  @app.get("/")
  def get(self):
    return f"Welcome to the model zoo serving system."
  
  @app.get("/utility")
  def get(self):
    return f"Current utility score: {self.utility}"

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
    
  @app.post("/classify_")
  async def classify_(self, model_name:str, file: UploadFile = File(...)):
    self.validate(model_name)
    
    image_bytes = await file.read()
    return self.route_request(model_name, image_bytes)
  

  def s_route_request(self, accuracy: float, latency: float, image_payload_bytes: bytes):
    """
    Securely Rrute the classification request to the appropriate model deployment,
    based on PSML defense algorithm 
    """
    query = (accuracy, latency)
    selected = self.proxy.l1_permute_and_flip_mechanism(query) # selected: (accuracy, latency)
    model_name = self.proxy.m_query(selected)
    
    self.utility += self.proxy.l1_score(float(selected[0]), float(selected[1]), query[0], query[1])

    print("s_route_request: {}", model_name)

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
    

  @app.post("/secure_classify_")
  async def secure_classify_(self, accuracy: float, latency: float, file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    return self.s_route_request(accuracy, latency, image_bytes)

def builder(args: Dict[str, str]) -> Application:
    """
    Build and return the Ray Serve Application based on arguments from `config.yaml`.
    """

    dataset = args.get("dataset", "cifar10")
    eps = float(args.get("eps", "0.1"))

    print(f"Building deployment with dataset={dataset}")
    print(f"eps: {eps}")

    return Router.bind(dataset, eps)
