CUDA_VISIBLE_DEVICES='0' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 0 --debug &> logs/log_0 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 1 --debug &> logs/log_1 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 2 --debug &> logs/log_2 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 3 --debug &> logs/log_3 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 4 --debug &> logs/log_4 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 5 --debug &> logs/log_5 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 6 --debug &> logs/log_6 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 7 --debug &> logs/log_7 &
wait;
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 8 --debug &> logs/log_8 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 9 --debug &> logs/log_9 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 10 --debug &> logs/log_10 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 11 --debug &> logs/log_11 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 12 --debug &> logs/log_12 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 13 --debug &> logs/log_13 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 14 --debug &> logs/log_14 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --model vgg19 --epochs 100 --n_shadows 16 --shadow_id 15 --debug &> logs/log_15 &
wait;
