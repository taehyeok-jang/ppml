
1. [Experiment Settings](#experiment-settings)
   - 1.1 [CIFAR-10 (60K, 10 classes)](#cifar-10-60k-10-classes)
   - 1.2 [CIFAR-100 (60K, 100 classes)](#cifar-100-60k-100-classes)
   

# Experiment Settings 

The evaluation of membership inference attacks will **be consistent** between, 
- the experimental settings used in the original paper (online and offline) and
- the modified settings (an 80-20 split for train-eval, with only offline attacks). 

In the offline attack scenario, each data point (x,y) is exclusively evaluated with (used for computing confidence scores) only shadow models that have not been seen these points during training. Consequently, no in-distribution data (of shadow models) is used for evaluation.

Therefore, we **expect the similar results** across both experimental settings. 



## CIFAR-10 (60K, 10 classes)

### the experimental settings (LiRA paper)

![fprtpr_efficientnet_b7_cifar10_asis](https://github.com/user-attachments/assets/4ad33867-9d24-49ea-ace2-cfea624ff0af)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (online)                   | 0.6588 | 0.5949   | 0.0295       |
| Ours (online, fixed variance)   | 0.6878 | 0.6176   | 0.0343       |
| Ours (offline)                  | 0.5750 | 0.5731   | 0.0165       |
| Ours (offline, fixed variance)  | 0.5847 | 0.5678   | 0.0124       |
| Global threshold                | 0.5809 | 0.5817   | 0.0012       |


### 80-20 offline attack 

![fprtpr_efficientnet_b7_cifar10_80-20_offline_attack](https://github.com/user-attachments/assets/97d43ea0-9ec7-4425-96d0-009b78a07548)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                  | 0.5713 | 0.5566   | 0.0006       |
| Ours (offline, fixed variance)  | 0.5664 | 0.5551   | 0.0006       |
| Global threshold                | 0.5629 | 0.5723   | 0.0004       |


## CIFAR-100 (60K, 100 classes)

### the experimental settings (LiRA paper)

![fprtpr_efficientnet_b7_cifar100_asis](https://github.com/user-attachments/assets/042be15c-4a23-4b3e-82f6-f5a07295c561)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (online)                   | 0.8449 | 0.7504   | 0.0764       |
| Ours (online, fixed variance)   | 0.8527 | 0.7594   | 0.0728       |
| Ours (offline)                  | 0.7249 | 0.6693   | 0.0251       |
| Ours (offline, fixed variance)  | 0.7142 | 0.6664   | 0.0094       |
| Global threshold                | 0.6882 | 0.6615   | 0.0013       |


### 80-20 offline attack 

![fprtpr_efficientnet_b7_cifar100_80-20_offline_attack](https://github.com/user-attachments/assets/4a9b0094-9564-459f-977c-79ea3e430f66)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                  | 0.7899 | 0.7177   | 0.0897       |
| Ours (offline, fixed variance)  | 0.7773 | 0.7147   | 0.0367       |
| Global threshold                | 0.7126 | 0.6971   | 0.0008       |

