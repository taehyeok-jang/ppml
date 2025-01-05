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
#  'vit_b16',
#  'vit_b32',
#  'vit_h14',
#  'vit_l16',
#  'vit_l32'
 ]

MODEL_REPO_UPSTREAM = "chenyaofo/pytorch-cifar-models"

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class ModelServer:
  def __init__(self, model_name: str, dataset: str):
    self.count = 0
    
    self.model_name = model_name
    self.dataset = dataset
    print(f"🚀 Loading model: {model_name}, fine-tuned for {dataset}")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.device.type == "cuda":
      gpu_index = torch.cuda.current_device()  
      gpu_name = torch.cuda.get_device_name(gpu_index)
      print(f"Using GPU: {gpu_index} ({gpu_name})")

    self.model = self.load_model(model_name).to(self.device)

    self.preprocessor = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),  # remove the alpha channel
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])

  def load_model(self, model_name: str):
    if model_name in SUPPORTED_MODELS:
        model_id = f'{self.dataset}_{model_name}'
        model = torch.hub.load(MODEL_REPO_UPSTREAM, model_id, pretrained=True)
    else:
        raise ValueError(f"Model {model_id} not available.")
    
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
    model_name = args.get("model_name", "resnet20")
    dataset = args.get("dataset", "cifar10")
    print(f"Building deployment with model_name={model_name}, dataset={dataset}")

    return ModelServer.bind(model_name, dataset)