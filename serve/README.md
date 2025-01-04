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
