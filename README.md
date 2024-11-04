# ppml


## Attacks

### Membership Inference Attacks 

- approches
  - higher confidence (prob distribution of predictions)
  - label-only 

- system architecture 
  - multiple shadow models (Shokri et al., 2016)
  - a single shadow model 


| Year   | Title |  Adversarial Knowledge | Target Model  |   Venue  | Paper Link  | Code Link |
|-------|--------|--------|--------|-----------|------------|---------------|
| 2024 | **Low-Cost High-Power Membership Inference Attacks** | Black-box | Classification Models | ICML | [Link](https://openreview.net/pdf?id=sT7UJh5CTc) | [Link](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia) |
| 2024 | **Scalable Membership Inference Attacks via Quantile Regression** | Black-box | Classification Models | NeurIPS | [Link](https://assets.amazon.science/61/10/fe6e935b49bf89bb34dded96a17b/scalable-membership-inference-attacks-via-quantile-regression.pdf) | [Link](https://github.com/amazon-science/quantile-mia) |
| 2021 | **Label-only membership inference attacks** | Black-box | Classification Models | ICML | [Link](http://proceedings.mlr.press/v139/choquette-choo21a.html) | [Link](https://github.com/cchoquette/membership-inference) |
| 2021 | **Membership Inference Attacks From First Principles** | White-box; Black-box | Classification Models | S&P | [Link](https://arxiv.org/abs/2112.03570) | [Link](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021) |
| 2019 | **ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models** | Black-box | Classification Models | NDSS | [Link](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-1_Salem_paper.pdf) | [Link](https://github.com/AhmedSalem2/ML-Leaks) |
| 2018 | **Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting** | Black-box | Classification Models | CSF | [Link](https://ieeexplore.ieee.org/abstract/document/8429311?casa_token=NQu6-mEb9JMAAAAA:LTU3BPSYc8ALHF89ifdWs1zl__ABgBzIr44xFoN2t8HwjTb5vm20S00VeH9JSmaBU-miBt5Ucg) | [Link](https://github.com/samuel-yeom/ml-privacy-csf18) |
| 2017  | **Membership inference attacks against machine learning models** | Black-box | Classification Models | S&P | [link](https://ieeexplore.ieee.org/abstract/document/7958568?casa_token=YOmVjvUemFUAAAAA:gGeuARxnjASvh9gnPkijkLD7d7HD1VV1JZkooXtS6tb6LGfKqHgBbyoaI-0-X7kFeP-3bjUR2A) | [Link1](https://github.com/yonsei-sslab/MIA) [Link2](https://github.com/csong27/membership-inference) |


**Low-Cost High-Power Membership Inference Attacks (2024)**

![ml_rmia](https://github.com/user-attachments/assets/9ebae5b6-1ba7-4249-9743-1565ae855575)

**Scalable Membership Inference Attacks via Quantile Regression (2024)**

<!--
![ml_quantile_reg](https://github.com/user-attachments/assets/9b6ba635-8903-4ca2-9a0b-d06df794c89c)
--> 

**Membership Inference Attacks From First Principles (2021)**

<img src="https://github.com/user-attachments/assets/876879b1-0eb0-411e-9c4a-28aa698f968a" width="600"/>
<br>

<img src="https://github.com/user-attachments/assets/66ff3c16-9901-4146-ba41-04126068eed5" width="600"/>
<img src="https://github.com/user-attachments/assets/d99556bb-ab39-4857-a69f-700658c5e139" width="600"/>
<br>

<img src="https://github.com/user-attachments/assets/503b02bf-f933-4bda-aa66-1c376dd6d029" width="600"/>


**Label-only membership inference attacks (2021)**

<!--
![ml_label-only_attack](https://github.com/user-attachments/assets/2b502d63-08e8-4f2b-8a43-acbbfd2c9755)
--> 

**ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models (2019)**

<!--
![ml_leaks](https://github.com/user-attachments/assets/9302aa65-f842-42f4-87f7-13edcd7f1345)
--> 

**Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting (2018)**

<!--
![ml_gap_attack](https://github.com/user-attachments/assets/d1e819c1-4ca9-4b40-8fd6-2538e8ee271f)
--> 

**Membership inference attacks against machine learning models (2017)**

![mia_fig-2](https://github.com/user-attachments/assets/95040234-984a-4ccf-b56a-0a822fd9de1d)

![mia_fig-3](https://github.com/user-attachments/assets/3feb27fa-5473-4eda-92fb-45eefc3d3677)



<!--
![ml_mia](https://github.com/user-attachments/assets/637e7480-3c44-4640-baea-a1e5db1c88dc)

![ml_mia_](https://github.com/user-attachments/assets/e13472f0-4182-4e1f-9448-aef9ba6ac0e1)
--> 

## *Assumption for Membership Inference Attack

### 1 Data Distribution

**Training Datasets**

- D”: the entire data distribution 
- D ~ D” (D ⊂ ~D”) (**victim model** is trained on D)
- D_shadow ~ D" (D_shadow ⊂ ~D”)

e.g. in our experiments, 

- D”: ImageNet (14.19M image, >20K classes, the overall distribution of the images of objects in the world) 
- D: imagenet-1k (1.28M images, 1K classes) 


(from Shokri, 2017) the paper assumes that

**(target-shadow)**
1. ∀i, D_train_shadow_i ∩ D_train_target = ∅ (worst case)
2. these datasets are also allowed to have some overlapped, (perform better)

**(shadow-shadow)**
1. ∀i, D_train_shadow_i (IN) ∩ D_test_shadow_i (OUT) = ∅
2. D_train_shadow_i ∩ D_train_shadow_j != ∅ (can be overlapped)
3. |D_train_shadow_i| ~= |D_test_shadow_i|

(from Shokri, 2017) For generating training data for shadow models, 
1. Model-based synthesis (no prior knowledge about the distribution of the target’s training dataset) 
2. Statistics-based systhesis
3. Noisy real data 


- each shadow model is trained from samples on, 
  - Q_in(x, y) =	{f ← T (D ∪ {(x, y)}) | D <- D”}
  - Q_out(x, y) =	{f ← T (D \ {(x, y)}) | D <- D”}


### 2 Model

(from Shokri, 2017) the paper assumes that the attacker either 
1. know the type and architecture of the ML model, as well as training algorithm
2. has black-box access to a ML model*

**In out experiments,** we assume that victim model arch ~= shadow model arch

1. victim model: pretrained on ImageNet 
2. shadow model: 
    1. model from scratch: difficult to train (resource consuming, poor performance)
    2. pretrained model: share in-distribution data with victim model 


### How to Set Up Our Experimental Environments? 

=> depends on the required settings. 

1. our model returns,
    - only a class label 
    - top k 
    - prob distributions
2. defense mechanism
    1. our models are overfitted, 
    2. our systems perturb a model’s predictions during inference 
3. adversary’s goals 
    1. computational efficiency (num of shadow models, num of queries to target model)  vs
    2. accuracy, precision/recall
  


# References

@article{hu2022membership,
  title={Membership inference attacks on machine learning: A survey},
  author={Hu, Hongsheng and Salcic, Zoran and Sun, Lichao and Dobbie, Gillian and Yu, Philip S and Zhang, Xuyun},
  journal={ACM Computing Surveys (CSUR)},
  volume={54},
  number={11s},
  pages={1--37},
  year={2022},
  publisher={ACM New York, NY}
}
