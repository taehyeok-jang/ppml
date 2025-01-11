import os
import torch
import torch.nn as nn
import timm


from huggingface_hub import HfApi, HfFolder, Repository, CommitOperationAdd
from huggingface_hub import create_repo
from huggingface_hub import upload_folder


# 📂 **환경 설정**
MODEL_NAME = "convnext_large"
DATASET = "cifar100"

CHECKPOINT_PATH = f'./vision-transformers-cifar10/checkpoint/{DATASET}/{MODEL_NAME}-4-ckpt.t7'
MODEL_REPO = f'tjang31/{MODEL_NAME}-{DATASET}'
SAVE_DIRECTORY = 'hf_model'

if DATASET == "cifar10":
    n_classes = 10 
elif DATASET == "cifar100":
    n_classes = 100
else: 
    raise ValueError(f"Unsupported dataset: {DATASET}")


# # 1️⃣ **Hugging Face 저장소 생성**
print("==> Creating Hugging Face repository..")
create_repo(repo_id=MODEL_REPO, private=False, exist_ok=True)

# 2️⃣ **Checkpoint 불러오기**
print("==> Loading checkpoint..")
checkpoint = torch.load(CHECKPOINT_PATH)
state_dict = checkpoint['model']
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

# 3️⃣ **모델 초기화 및 가중치 적용**
print("==> Loading model with timm..")
model = timm.create_model(MODEL_NAME, pretrained=True)

if MODEL_NAME.startswith("vit"):
    model.head = nn.Linear(model.head.in_features, n_classes)
elif MODEL_NAME.startswith("convnext"):
    model.head.fc = nn.Linear(model.head.fc.in_features, n_classes)
else: 
    raise ValueError(f"Unsupported network: {MODEL_NAME}")

print(model)

model.load_state_dict(state_dict)

# # 4️⃣ **로컬 저장**
print("==> Saving model locally..")
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")

os.makedirs(SAVE_DIRECTORY, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIRECTORY, 'pytorch_model.bin'))

# 모델 카드 및 config 파일 생성
with open(os.path.join(SAVE_DIRECTORY, 'config.json'), 'w') as f:
    f.write(f'{{"architecture": "{MODEL_NAME}", "num_classes": {n_classes}}}')

# 5️⃣ **Hugging Face Hub에 업로드**
print("==> Uploading model to Hugging Face Hub..")

upload_folder(
    folder_path=SAVE_DIRECTORY,
    repo_id=MODEL_REPO,
    commit_message="Upload fine-tuned model"
)


print(f"Model successfully uploaded to https://huggingface.co/{MODEL_REPO}")