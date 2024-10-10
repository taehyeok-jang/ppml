#!/bin/bash
# compares RMIA and prior works on multiple datasets using 1 OUT reference model
target_idx=0 # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10" "cifar100" "cinic10" "purchase100")

for dataset in "${datasets[@]}";
do
    ### 1 ref model
    python main.py --cf "attack_configs/${dataset}/attack_P.yaml" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
done

# plotting using matplotlib
python plot.py --figure 3