import torch
from torchvision import transforms
from pprint import pprint

from PIL import Image

print('from https://github.com/chenyaofo/pytorch-cifar-models... ')
pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True, force_reload=True)
print('vgg19_bn: ')
print(model)

preprocessor = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),  # remove the alpha channel
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])

image = Image.open("dataset/grey-British-Shorthair-compressed.jpg").convert("RGB")
preprocessed_image = preprocessor(image).unsqueeze(0)
print('preprocessed_image.shape: ', preprocessed_image.shape)  # Expected output: torch.Size([1, 3, 224, 224])

with torch.no_grad():
    output = model(preprocessed_image)
    predicted_class = torch.argmax(output[0])
    print('predicted_class: ', predicted_class)