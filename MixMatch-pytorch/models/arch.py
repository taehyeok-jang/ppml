import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm


def network(arch: str, pretrained: bool, n_classes=None):
    print(f'arch: {arch}, pretrained: {pretrained}, n_classes: {n_classes}') 

    if n_classes is None:
        raise ValueError("n_classes must be specified and greater than 0.")
    
    # https://pytorch.org/vision/stable/models.html
    TORCHVISION_MODELS = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'densenet121', 
                          'wide_resnet50_2', 'wide_resnet101_2',
                          'densenet201', 'mobilenet_v2', 'inception_v3', 
                          'efficientnet_b0', 'efficientnet_b7', 
                          'squeezenet1_0', 'alexnet', 'googlenet', 'shufflenet_v2_x1_0']
    
    # https://github.com/huggingface/pytorch-image-models
    PYTORCH_IMAGE_MODELS = ['vit_base_patch16_224', 'vit_large_patch16_224', 'deit_base_patch16_224',
                        'convnext_base', 'convnext_large']
    
    if arch in TORCHVISION_MODELS:
        model = models.__dict__[arch](pretrained=pretrained)
    elif arch in PYTORCH_IMAGE_MODELS:
        model = timm.create_model(arch, pretrained=pretrained)
    else:
        raise ValueError(f"Model {arch} not available.")
    
    if arch == "vgg19": # for VGG-19
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, n_classes)  
    elif arch in ["wide_resnet50_2", "wide_resnet101_2"]:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
    elif arch == "vit_large_patch16_224": # for ViT (vision transformer) 
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, n_classes)        
    elif arch == "efficientnet_b7": # for efficientnet
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
    else:
        raise ValueError(f"Model {arch} not supported yet.")

    freeze_interdemidate_layers(model, arch)
        
    return model


def freeze_interdemidate_layers(model, arch: str):
    """
    Freeze specific layers of a model for efficient fine-tuning.
    
    Parameters:
    - model (torch.nn.Module): The model to apply freezing to.
    - arch (str): The name of the model.
    """
    #if arch == "vgg19":
    #    print("Freezing VGG-19 intermediate layers...")
    #    for param in model.features.parameters():
    #        param.requires_grad = False
    
    if arch == "vit_large_patch16_224":
        print("Freezing ViT-Large intermediate layers...")
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    
    #elif arch == "efficientnet_b7":
    #    print("Freezing EfficientNet-B7 intermediate layers...")
    #    for param in model.parameters():
    #        param.requires_grad = False
    #    for param in model.classifier[1].parameters():
    #        param.requires_grad = True
    
    else:
        print(f"Do not freeze layers for model: {arch}")
    
    return model