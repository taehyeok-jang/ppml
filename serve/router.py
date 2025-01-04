

''' TODO 

1. implement router. 
    - route (call application (or deployment)) base on 'model_name' in the request params.
    - this logic will be replaced by pareto-front profiled set, combined with defense

2. advancements 
    - batch inference (8+ images per request)

3. model zoo for specific dataset
    - CIFAR-10
    - CIFAR-100

4. resource allocation w/ monitoring  
    - if only cpu are used for serving, then experiments are significantly slow. 
    - monitor the system resource usages, and use multi gpus for inference serving. 

    
'''