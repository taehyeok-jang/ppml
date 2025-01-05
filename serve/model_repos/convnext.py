import torch
from torchvision import transforms
from pprint import pprint

from PIL import Image

import torch
from transformers import AutoModel

''' RESULT
tiny: 27820128
base: 87566464

anonauthors/cifar10-timm-convnext_base.fb_in1k: 27820128 parameters
anonauthors/cifar10-ConvNeXt-base: 87566464 parameters

karan99300/ConvNext-finetuned-CIFAR100: 27820128 parameters
anonauthors/cifar100-timm-convnext_base.fb_in1k: 27820128 parameters
anonauthors/cifar100-ConvNeXt-base: 87566464 parameters
'''

# 모델 체크포인트 리스트
model_checkpoints = [
    "anonauthors/cifar10-timm-convnext_base.fb_in1k",
    "anonauthors/cifar10-ConvNeXt-base",
    "karan99300/ConvNext-finetuned-CIFAR100",
    "anonauthors/cifar100-timm-convnext_base.fb_in1k",
    "anonauthors/cifar100-ConvNeXt-base",
]

# 모델 파라미터 수 확인
for checkpoint in model_checkpoints:
    try:
        model = AutoModel.from_pretrained(checkpoint)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{checkpoint}: {total_params} parameters")
    except Exception as e:
        print(f"{checkpoint}: Failed to load - {e}")
