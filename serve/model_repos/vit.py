import torch
from torchvision import transforms
from pprint import pprint

from PIL import Image

import torch
from transformers import AutoModel

''' RESULT
small: 21,813,504
base: 86,389,248
large: 304,351,232

MF21377197/vit-small-patch16-224-finetuned-Cifar10: 					21813504 parameters
raks87/vit-small-patch16-224-finetuned-cifar10: 						21813504 parameters
aaraki/vit-base-patch16-224-in21k-finetuned-cifar10: 					86389248 parameters
02shanky/vit-finetuned-cifar10: 										86389248 parameters

Ahmed9275/Vit-Cifar100: 												86389248 parameters
edumunozsala/vit_base-224-in21k-ft-cifar100: 							86389248 parameters
edadaltocg/vit_base_patch16_224_in21k_ft_cifar100: 						86389248 parameters
jialicheng/cifar100-vit-large: 											304351232 parameters
'''

# 모델 체크포인트 리스트
model_checkpoints = [
    "MF21377197/vit-small-patch16-224-finetuned-Cifar10",
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "02shanky/vit-finetuned-cifar10",
    "raks87/vit-small-patch16-224-finetuned-cifar10",
    "Ahmed9275/Vit-Cifar100",
    "jialicheng/cifar100-vit-large",
    "edumunozsala/vit_base-224-in21k-ft-cifar100",
    "edadaltocg/vit_base_patch16_224_in21k_ft_cifar100"
]

# 모델 파라미터 수 확인
for checkpoint in model_checkpoints:
    try:
        model = AutoModel.from_pretrained(checkpoint)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{checkpoint}: {total_params} parameters")
    except Exception as e:
        print(f"{checkpoint}: Failed to load - {e}")
