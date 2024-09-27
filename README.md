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


**Low-Cost High-Power Membership Inference Attacks (2024)**

![ml_rmia](https://github.com/user-attachments/assets/9ebae5b6-1ba7-4249-9743-1565ae855575)

**Scalable Membership Inference Attacks via Quantile Regression (2024)**

![ml_quantile_reg](https://github.com/user-attachments/assets/9b6ba635-8903-4ca2-9a0b-d06df794c89c)


**Membership Inference Attacks From First Principles (2021)**

**Label-only membership inference attacks (2021)**

![ml_label-only_attack](https://github.com/user-attachments/assets/2b502d63-08e8-4f2b-8a43-acbbfd2c9755)

**ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models (2019)**

![ml_leaks](https://github.com/user-attachments/assets/9302aa65-f842-42f4-87f7-13edcd7f1345)

**Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting (2018)**

![ml_gap_attack](https://github.com/user-attachments/assets/d1e819c1-4ca9-4b40-8fd6-2538e8ee271f)



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
