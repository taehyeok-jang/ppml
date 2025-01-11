import torch
import timm
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 📂 **1️⃣ 모델 다운로드 및 로드**
MODEL_NAME = "convnext_base"
DATASET = "cifar100"
MODEL_REPO = f'tjang31/{MODEL_NAME}-{DATASET}'

if DATASET == "cifar10":
    n_classes = 10 
elif DATASET == "cifar100":
    n_classes = 100
else: 
    raise ValueError(f"Unsupported dataset: {DATASET}")

# Hugging Face에서 모델 state_dict 다운로드
# state_dict = torch.hub.load_state_dict_from_url(
#     f'https://huggingface.co/{MODEL_REPO}/resolve/main/pytorch_model.bin',
#     #f'https://huggingface.co/{MODEL_REPO}/resolve/main/pytorch_model.bin',
#     map_location='cpu'
# )
checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename="pytorch_model.bin")
state_dict = torch.load(checkpoint_path, map_location='cpu')

# print('state_dict:')
# for key, value in state_dict.items():
#     print(f"{key}: {value.shape}")

# 모델 초기화
model = timm.create_model(MODEL_NAME, pretrained=False)
if MODEL_NAME.startswith("vit"):
    model.head = nn.Linear(model.head.in_features, n_classes)
elif MODEL_NAME.startswith("convnext"):
    model.head.fc = nn.Linear(model.head.fc.in_features, n_classes)
else: 
    raise ValueError(f"Unsupported network: {MODEL_NAME}")
model.load_state_dict(state_dict)

model.eval()

# 📂 **2️⃣ 이미지 전처리**
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 테스트 이미지 불러오기
image_path = '../dataset/grey-British-Shorthair-compressed.jpg'

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

# 📂 **3️⃣ 모델 예측**
with torch.no_grad():
    output_tensor = model(input_tensor)
    # predictions = torch.softmax(output_tensor, dim=1)
    # predicted_class = predictions.argmax(dim=1).item()
    # confidence = predictions.max().item()

    predictions = torch.softmax(output_tensor, dim=1).squeeze(0).cpu().numpy()  # 1-D array로 변환

    # 가장 높은 확률 클래스 예측
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    print('output_tensor: ', output_tensor)
    print('predictions: ', predictions)
    print('predicted_class: ', predicted_class)
    print('confidence: ', confidence)



print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")