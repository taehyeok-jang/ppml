import ray
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, UploadFile, File
import numpy as np 

from typing import List
from PIL import Image

from typing import Dict
from io import BytesIO

import requests

from model_profiles import (
    SUPPORTED_MODELS, 
    CIFAR10_PARETO_FRONT_MODELS,
    CIFAR10_PARETO_FRONT_SPEC,
    CIFAR100_PARETO_FRONT_MODELS,
    CIFAR100_PARETO_FRONT_SPEC
)

from psml_defense import PsmlDefenseProxy, PsmlDefenseProxyV2


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
  def __init__(self, dataset: str, eps: float):
    """
    eps:
        only used to initialize phase, in PsmlDefenseProxy(V1)
    """

    self.count = 0
    if dataset == "cifar10":
        pareto_front_models = CIFAR10_PARETO_FRONT_MODELS
        pareto_front_spec = CIFAR10_PARETO_FRONT_SPEC
    elif dataset == "cifar100":
        pareto_front_models = CIFAR100_PARETO_FRONT_MODELS
        pareto_front_spec = CIFAR100_PARETO_FRONT_SPEC
    else: 
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    self.proxy = PsmlDefenseProxyV2(
        pareto_front_models=pareto_front_models, 
        pareto_front_spec=pareto_front_spec, 
        sensitivity=0.01
       )
    
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
    Securely Route the classification request to the appropriate model deployment
    """

    query = (accuracy, latency)
    selected = self.proxy.l1_permute_and_flip_mechanism(query) # selected: (accuracy, latency)
    model_name = self.proxy.m_query(selected)
    
    self.utility += self.proxy.l1_score(float(selected[0]), float(selected[1]), query[0], query[1])

    print("s_route_request for {}, query {}", model_name, query)

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


  def s_route_request_batch(self, accuracy: float, latency: float, eps: float, image_payloads: List[bytes]):
      """
      Securely Route the classification request to the appropriate model deployment for a batch of image payloads.
      """
      query = (accuracy, latency)
      selected = self.proxy.l1_permute_and_flip_mechanism(eps, query)  # selected: (accuracy, latency)
      model_name = self.proxy.m_query(selected)
      
      self.utility += self.proxy.l1_score(float(selected[0]), float(selected[1]), query[0], query[1])
      print(f"s_route_request_batch to {model_name}, query {query} - eps {eps}")

      model_endpoint = f"http://localhost:8000/v2/{model_name}/b_classify_"
      try:
          files = [("files", ("image.jpg", BytesIO(img), "image/jpeg")) for img in image_payloads]
          response = requests.post(model_endpoint, files=files)
          if response.status_code == 200:
              return response.json()
          else:
              return {"error": f"Failed to query model {model_name}. HTTP {response.status_code}"}
      except Exception as e:
          return {"error": str(e)}


  @app.post("/b_secure_classify_")
  async def batch_secure_classify_(
      self, accuracy: float, latency: float, eps: float, files: List[UploadFile] = File(...)
  ):
      """
      Secure classification for a batch of images.
      """
      image_payloads = [await file.read() for file in files]
      return self.s_route_request_batch(accuracy, latency, eps, image_payloads)


def builder(args: Dict[str, str]) -> Application:
    """
    Build and return the Ray Serve Application based on arguments from `config.yaml`.
    """

    dataset = args.get("dataset", "cifar10")
    eps = float(args.get("eps", "0.1"))
  
    print(f"Building deployment with dataset={dataset}")
    print(f"eps: {eps}")

    return Router.bind(dataset, eps)
