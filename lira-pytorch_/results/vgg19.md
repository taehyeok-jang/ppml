
# Experiment Settings 

The evaluation of membership inference attacks will **be consistent** between, 
- the experimental settings used in the original paper (online and offline) and
- the modified settings (an 80-20 split for train-eval, with only offline attacks). 

In the offline attack scenario, each data point (x,y) is exclusively evaluated with (used for computing confidence scores) only shadow models that have not been seen these points during training. Consequently, no in-distribution data (of shadow models) is used for evaluation.

Therefore, we **expect the similar results** across both experimental settings. 



## CIFAR-10 (60K, 10 classes)

### the experimental settings (LiRA paper)

![fprtpr_vgg19_asis](https://github.com/user-attachments/assets/6d502c45-2e9a-4c33-8cb9-4c99edc6909d)

| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (online)                 | 0.6801 | 0.6117   | 0.0423      |
| Ours (online, fixed variance) | 0.6781 | 0.6097   | 0.0366      |
| Ours (offline)                | 0.5252 | 0.5344   | 0.0097      |
| Ours (offline, fixed variance)| 0.5343 | 0.5369   | 0.0111      |
| Global threshold              | 0.5583 | 0.5688   | 0.0010      |

### 80-20 offline attack 

![fprtpr_vgg19_80-20_offline_attack](https://github.com/user-attachments/assets/ab6a6f6c-5d34-451b-8c62-97603707c716)

| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                | 0.5609 | 0.5607   | 0.0246      |
| Ours (offline, fixed variance)| 0.5745 | 0.5608   | 0.0227      |
| Global threshold              | 0.5599 | 0.5717   | 0.0012      |


## CIFAR-100 (60K, 100 classes)

### the experimental settings (LiRA paper)

![fprtpr_vgg19_cifar100_asis](https://github.com/user-attachments/assets/b1b8cc0d-44fd-475f-995f-30a1e5871df8)

| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (online)                 | 0.8600 | 0.7600   | 0.1025      |
| Ours (online, fixed variance) | 0.8580 | 0.7620   | 0.0782      |
| Ours (offline)                | 0.6707 | 0.6491   | 0.0289      |
| Ours (offline, fixed variance)| 0.6765 | 0.6402   | 0.0221      |
| Global threshold              | 0.6796 | 0.6631   | 0.0015      |

### 80-20 offline attack 

![fprtpr_vgg19_cifar100_80-20_offline_attack](https://github.com/user-attachments/assets/ca9be04b-a093-4c42-8a72-3c43b1c99658)


| Attack                        | AUC    | Accuracy | TPR@0.1%FPR |
|-------------------------------|--------|----------|-------------|
| Ours (offline)                | 0.6717 | 0.6513   | 0.0229      |
| Ours (offline, fixed variance)| 0.6800 | 0.6448   | 0.0395      |
| Global threshold              | 0.6618 | 0.6488   | 0.0010      |


