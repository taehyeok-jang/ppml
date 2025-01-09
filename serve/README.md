## INSTALLING

```
pip install -U "ray[data,train,tune,serve]"
pip install -U "ray[default]"

pip install transformers requests torch
```


## RUNNING THE CODE

```script
### ray cluster
ray start --head 
# ray stop 

### ray serve 
serve build image_classifier:app text_translator:app -o config.yaml
serve deploy config.yaml

```

## TODO  

priority: 
    - [x] monitor metrics (e.g. num of requests to each model, system resource usages) => let's use prometheus,  since grafana installation is restrited due to limited authorization. 
    - [] CIFAR-10
    - [] this logic will be replaced by pareto-front profiled set, combined with defense
    - [] batch inference (e.g. 8+ images per request)

==== 

1. implement router. 
    - [x] route (call application (or deployment)) base on 'model_name' in the request params.
    - [] this logic will be replaced by pareto-front profiled set, combined with defense

2. advancements 
    - [] batch inference (e.g. 8+ images per request)

3. model zoo for specific dataset
    - [] CIFAR-10
    - [] CIFAR-100

4. resource allocation w/ monitoring  
    - [] use multi gpus (cuda:0~3 only among 8 A40) # code 이거 왜 gpu:0 으로만 세팅되냐...
    - [] monitor metrics (e.g. num of requests to each model, system resource usages), 
 