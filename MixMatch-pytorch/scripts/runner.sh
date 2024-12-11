model='vgg19'
gpu_ids=(0 1 2 3 4 5 6 7)
n_labeled=10000
lr_=(0.01 0.005)
lambda_u_=(10 25 75 100)
alpha=0.75

combinations=()
for lr in "${lr_[@]}"; do
  for lambda_u in "${lambda_u_[@]}"; do
    combinations+=("$lr $lambda_u")
  done
done

for i in "${!combinations[@]}"; do
  gpu_id=${gpu_ids[$((i % ${#gpu_ids[@]}))]} 
  combo=(${combinations[$i]})
  lr=${combo[0]}
  lambda_u=${combo[1]}
  
  nohup python -u train.py --gpu ${gpu_id} --model ${model} --n-labeled ${n_labeled} \
  --lr ${lr} --alpha ${alpha} --lambda-u ${lambda_u} \
  --out ${model}/cifar10@${n_labeled}_alpha_${alpha}_lr_${lr}_lambda_u_${lambda_u} \
  --debug True \
  > logs/${model}_alpha_${alpha}_lr_${lr}_lambda_u_${lambda_u}.log 2>&1 &
done