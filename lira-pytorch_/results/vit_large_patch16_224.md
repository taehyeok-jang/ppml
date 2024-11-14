
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

![fprtpr_vit_large_patch16_224_cifar10_asis](https://github.com/user-attachments/assets/3a2e8b07-1ea8-419e-9444-3086bf9ec004)


| Attack                          | AUC   | Accuracy | TPR@0.1% FPR |
|---------------------------------|-------|----------|--------------|
| Ours (online)                   | 0.5273 | 0.5164   | 0.0123       |
| Ours (online, fixed variance)   | 0.5234 | 0.5141   | 0.0116       |
| Ours (offline)                  | 0.5089 | 0.5116   | 0.0072       |
| Ours (offline, fixed variance)  | 0.5076 | 0.5104   | 0.0049       |
| Global threshold                | 0.5025 | 0.5085   | 0.0007       |


### 80-20 offline attack 

![fprtpr_vit_large_patch16_224_cifar10_80-20_offline_attack](https://github.com/user-attachments/assets/c412e096-9b9a-4ae9-b2a9-a787eddb343d)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                  | 0.5034 | 0.5092   | 0.0012       |
| Ours (offline, fixed variance)  | 0.5088 | 0.5091   | 0.0018       |
| Global threshold                | 0.5078 | 0.5125   | 0.0002       |


## CIFAR-100 (60K, 100 classes)

### the experimental settings (LiRA paper)
![fprtpr_vit_large_patch16_224_cifar100_asis](https://github.com/user-attachments/assets/a53277b1-6127-4a33-920e-7dc07ae844de)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|---------------------------------|-------|----------|--------------|
| Ours (online)                   | 0.6789 | 0.6055   | 0.0775       |
| Ours (online, fixed variance)   | 0.6801 | 0.6083   | 0.0805       |
| Ours (offline)                  | 0.5796 | 0.5676   | 0.0401       |
| Ours (offline, fixed variance)  | 0.5844 | 0.5678   | 0.0501       |
| Global threshold                | 0.5336 | 0.5514   | 0.0016       |

### 80-20 offline attack 
![fprtpr_vit_large_patch16_224_cifar100_80-20_offline_attack](https://github.com/user-attachments/assets/0c48a7c1-4fc0-45a3-96c9-793b895ce1d6)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                  | 0.5353 | 0.5344   | 0.0074       |
| Ours (offline, fixed variance)  | 0.5382 | 0.5360   | 0.0160       |
| Global threshold                | 0.5289 | 0.5485   | 0.0004       |



