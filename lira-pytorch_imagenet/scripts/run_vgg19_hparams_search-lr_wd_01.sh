CUDA_VISIBLE_DEVICES='0' python3 -u train.py --model vgg19 --epochs 20 --lr 0.01 --weight_decay 0.0001 --n_shadows 8 --shadow_id 0 --debug &> logs/log_0 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --model vgg19 --epochs 20 --lr 0.01 --weight_decay 0.0005 --n_shadows 8 --shadow_id 1 --debug &> logs/log_1 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --model vgg19 --epochs 20 --lr 0.01 --weight_decay 0.001 --n_shadows 8 --shadow_id 2 --debug &> logs/log_2 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --model vgg19 --epochs 20 --lr 0.01 --weight_decay 0.01 --n_shadows 8 --shadow_id 3 --debug &> logs/log_3 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --model vgg19 --epochs 20 --lr 0.005 --weight_decay 0.0001 --n_shadows 8 --shadow_id 4 --debug &> logs/log_4 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --model vgg19 --epochs 20 --lr 0.005 --weight_decay 0.0005 --n_shadows 8 --shadow_id 5 --debug &> logs/log_5 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --model vgg19 --epochs 20 --lr 0.005 --weight_decay 0.001 --n_shadows 8 --shadow_id 6 --debug &> logs/log_6 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --model vgg19 --epochs 20 --lr 0.005 --weight_decay 0.01 --n_shadows 8 --shadow_id 7 --debug &> logs/log_7 &
