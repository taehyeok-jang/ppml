# Model Profiling 

| Model                  | Params | GFLOPS | Latency (*) | Note                  |
|------------------------|--------|--------|-------------|-----------------------|
| vgg-19                 | 143.7M | 19.63  | 0.002 ms    |                       |
| vit_large_patch16_224  | 303.3M | 59.7   | 0.009 ms    | 224x224 pixel images  |
| efficientnet_b7        | 66.3M  | 37.75  | 0.021 ms    |                       |


Latency: evaluated in our own _model zoo_ profiling experiment 


# Results 

## Online Attack

| Model                                    | CIFAR-10 | CIFAR-100 |
|------------------------------------------|----------|-----------|
| vgg-19                                    | 0.6117   | 0.76      |
| vit_large_patch16_224 (freeze intermediate layers) | 0.5164   | 0.6055    |
| efficientnet_b7                           | 0.5949   | 0.7504    |
| (baseline) ResNet                         | 0.638    |           |
| (baseline) Wide ResNet                    |          | 0.826     |


## Offline Attack 

**(paper)**

| Model                                    | CIFAR-10 | CIFAR-100 |
|------------------------------------------|----------|-----------|
| vgg-19                                    | 0.5344   | 0.6491    |
| vit_large_patch16_224 (freeze intermediate layers) | 0.5116   | 0.5676    |
| efficientnet_b7                           | 0.5731   | 0.6693    |
| (baseline) ResNet                         | N/A      |           |
| (baseline) Wide ResNet                    |          | N/A       |


## Offline Attack 

**(prposed experiment)**

| Model                                    | CIFAR-10 | CIFAR-100 |
|------------------------------------------|----------|-----------|
| vgg-19                                    | 0.5607   | 0.6513    |
| vit_large_patch16_224 (freeze intermediate layers) | 0.5092   | 0.5344    |
| efficientnet_b7                           | 0.5566   | 0.7177    |
| (baseline) ResNet                         | N/A      |           |
| (baseline) Wide ResNet                    |          | N/A       |

========

**Baseline Model (from LiRA paper)**
- ResNet:	92% test accuracy for half of CIFAR-10
- Wide ResNet:	60% test accuracy for half of CIFAR-100
