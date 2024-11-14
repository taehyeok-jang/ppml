CUDA_VISIBLE_DEVICES='0' python3 -u train.py --model resnet18 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 0 --debug &> logs/log_0 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --model resnet50 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 1 --debug &> logs/log_1 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --model resnet101 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 2 --debug &> logs/log_2 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --model wide_resnet50_2 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 3 --debug &> logs/log_3 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --model wide_resnet101_2 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 4 --debug &> logs/log_4 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --model efficientnet_b7 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 5 --debug &> logs/log_5 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --model vit_base_patch16_224 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 6 --debug &> logs/log_6 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --model vit_large_patch16_224 --epochs 50 --lr 0.001 --weight_decay 0.01 --n_shadows 8 --shadow_id 7 --debug &> logs/log_7 &
wait;

