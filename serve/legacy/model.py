import ray
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, UploadFile, File

import torch
from torchvision import transforms
import torchvision.models as models
import timm

from PIL import Image

from typing import Dict
from io import BytesIO


app = FastAPI()

# https://docs.ray.io/en/latest/serve/resource-allocation.html
# https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resource-requirements
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,  # Allocate 1 GPU per replica
        "runtime_env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}  # Restrict to GPU 0-3
    }
)
@serve.ingress(app)
class ModelServer:
  def __init__(self, model_name: str):
    self.count = 0
    
    self.model_name = model_name
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading model: {model_name}")
    if self.device.type == "cuda":
      gpu_index = torch.cuda.current_device()  
      gpu_name = torch.cuda.get_device_name(gpu_index)
      print(f"Using GPU: {gpu_index} ({gpu_name})")

    self.model = self.load_model(model_name).to(self.device)

    self.preprocessor = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),  # remove the alpha channel
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def load_model(self, model_name: str):
    if model_name in ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'densenet121',
                      'densenet201', 'mobilenet_v2', 'inception_v3', 'efficientnet_b0',
                      'efficientnet_b7', 'squeezenet1_0', 'alexnet', 'googlenet', 'shufflenet_v2_x1_0']:
        model = models.__dict__[model_name](pretrained=True)
    elif model_name in ['vit_base_patch16_224', 'vit_large_patch16_224', 'deit_base_patch16_224',
                        'convnext_base', 'convnext_large']:
        model = timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not available.")
    
    model.eval()
    return model


  def classify(self, image_payload_bytes):
    pil_image = Image.open(BytesIO(image_payload_bytes))

    pil_images = [pil_image]  #batch size is one
    input_tensor = torch.cat(
        [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        ).to(self.device)

    with torch.no_grad():
        output_tensor = self.model(input_tensor)
    return {"model": self.model_name, "class_index": int(torch.argmax(output_tensor[0]))}

  @app.get("/")
  def get(self):
      return f"Welcome to the {self.model_name} model serving system."

  @app.post("/classify_")
  async def classify_(self, file: UploadFile = File(...)):
    image_bytes = await file.read()
    return self.classify(image_bytes)

def app_builder(args: Dict[str, str]) -> Application:
    """
    Build and return the Ray Serve Application based on arguments from `config.yaml`.
    """
    model_name = args.get("model_name", "resnet18")
    print(f"Building deployment with model_name={model_name}")

    return ModelServer.bind(model_name)