import numpy as np 

CIFAR10_model_zoo_profile = {
  'mobilenetv2_x0_5': {'accuracy': 93.14, 'latency': 37.05},
  'mobilenetv2_x0_75': {'accuracy': 94.08, 'latency': 37.24},
  'mobilenetv2_x1_0': {'accuracy': 93.94, 'latency': 36.74},
  'mobilenetv2_x1_4': {'accuracy': 94.24, 'latency': 37.28},
  'repvgg_a0': {'accuracy': 94.32, 'latency': 41.06},
  'repvgg_a1': {'accuracy': 95.5, 'latency': 42.77},
  'repvgg_a2': {'accuracy': 95.22, 'latency': 43.98},
  'resnet20': {'accuracy': 92.68, 'latency': 17.45},
  'resnet32': {'accuracy': 93.7, 'latency': 26.64},
  'resnet44': {'accuracy': 93.92, 'latency': 33.15},
  'resnet56': {'accuracy': 94.26, 'latency': 44.26},
  'shufflenetv2_x0_5': {'accuracy': 90.6, 'latency': 44.63},
  'shufflenetv2_x1_0': {'accuracy': 93.74, 'latency': 46.28},
  'shufflenetv2_x1_5': {'accuracy': 93.7, 'latency': 45.31},
  'shufflenetv2_x2_0': {'accuracy': 93.68, 'latency': 47.42},
  'vgg11_bn': {'accuracy': 93.22, 'latency': 11.07},
  'vgg13_bn': {'accuracy': 93.72, 'latency': 12.83},
  'vgg16_bn': {'accuracy': 93.92, 'latency': 15.14},
  'vgg19_bn': {'accuracy': 94.3, 'latency': 18.17},
  'vit_small_patch16_384': {'accuracy': 98.02, 'latency': 35.86},
  'vit_base_patch16_384': {'accuracy': 98.32, 'latency': 36.27},
  'vit_large_patch16_384': {'accuracy': 98.38, 'latency': 67.79}
}


CIFAR100_model_zoo_profile = {
  'mobilenetv2_x0_5': {'accuracy': 71.04, 'latency': 36.91},
  'mobilenetv2_x0_75': {'accuracy': 74.42, 'latency': 37.05},
  'mobilenetv2_x1_0': {'accuracy': 74.44, 'latency': 36.12},
  'mobilenetv2_x1_4': {'accuracy': 76.04, 'latency': 37.0},
  'repvgg_a0': {'accuracy': 76.24, 'latency': 41.82},
  'repvgg_a1': {'accuracy': 76.44, 'latency': 41.98},
  'repvgg_a2': {'accuracy': 77.24, 'latency': 46.43},
  'resnet20': {'accuracy': 68.78, 'latency': 17.06},
  'resnet32': {'accuracy': 69.42, 'latency': 28.64},
  'resnet44': {'accuracy': 71.52, 'latency': 36.63},
  'resnet56': {'accuracy': 72.12, 'latency': 47.22},
  'shufflenetv2_x0_5': {'accuracy': 67.86, 'latency': 46.91},
  'shufflenetv2_x1_0': {'accuracy': 73.1, 'latency': 47.06},
  'shufflenetv2_x1_5': {'accuracy': 74.36, 'latency': 49.13},
  'shufflenetv2_x2_0': {'accuracy': 75.0, 'latency': 49.69},
  'vgg11_bn': {'accuracy': 70.44, 'latency': 10.97},
  'vgg13_bn': {'accuracy': 74.02, 'latency': 12.94},
  'vgg16_bn': {'accuracy': 74.68, 'latency': 16.6},
  'vgg19_bn': {'accuracy': 75.16, 'latency': 20.24},
  'vit_small_patch16_384': {'accuracy': 90.04, 'latency': 36.16},
  'vit_base_patch16_384': {'accuracy': 90.1, 'latency': 39.29},
  'vit_large_patch16_384': {'accuracy': 90.82, 'latency': 73.69}
}

PARETO_FRONT_MODELS = [
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'vit_small_patch16_384',
    'vit_base_patch16_384',
    'vit_large_patch16_384'
]

CIFAR10_PARETO_FRONT_SPEC = [
	(93.22, 11.07),
	(93.72, 12.83),
	(93.92, 15.14),
	(94.3 , 18.17),
	(98.02, 35.86),
	(98.32, 36.27),
	(98.38, 67.7)
]

CIFAR100_PARETO_FRONT_SPEC = [
	(70.44, 10.97),
	(74.02, 12.94),
	(74.68, 16.6 ),
	(75.16, 20.24),
	(90.04, 36.16),
	(90.1 , 39.29),
	(90.82, 73.69) 
]