mkdir lrft_finetuned

cd lrft_finetuned

mkdir 5k_exp1
mkdir 5k_exp2
mkdir 5k_exp3
mkdir 20k_exp1
mkdir 20k_exp2
mkdir 20k_exp3

cd ..

python train.py \
    --stage LRFT \
    --output-dir lrft_finetuned/5k_exp1 \
    --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
    --train-path data/lrft/kp20k_low-resource_5000_train_exp1.jsonl \
    --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
    --batch-size-train 16 \
    --batch-size-valid 8 \
    --max-learning-rate 1e-5 \
    --gpus 0 


# python train.py \
#     --stage LRFT \
#     --output-dir lrft_finetuned/5k_exp2 \
#     --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
#     --train-path data/lrft/kp20k_low-resource_5000_train_exp2.jsonl \
#     --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
#     --batch-size-train 16 \
#     --batch-size-valid 8 \
#     --max-learning-rate 1e-5 \
#     --gpus 0 


# python train.py \
#     --stage LRFT \
#     --output-dir lrft_finetuned/5k_exp3 \
#     --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
#     --train-path data/lrft/kp20k_low-resource_5000_train_exp3.jsonl \
#     --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
#     --batch-size-train 16 \
#     --batch-size-valid 8 \
#     --max-learning-rate 1e-5 \
#     --gpus 0 




python train.py \
    --stage LRFT \
    --output-dir lrft_finetuned/20k_exp1 \
    --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
    --train-path data/lrft/kp20k_low-resource_20000_train_exp1.jsonl \
    --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
    --batch-size-train 16 \
    --batch-size-valid 8 \
    --max-learning-rate 1e-5 \
    --gpus 0


# python train.py \
#     --stage LRFT \
#     --output-dir lrft_finetuned/20k_exp2 \
#     --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
#     --train-path data/lrft/kp20k_low-resource_20000_train_exp2.jsonl \
#     --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
#     --batch-size-train 16 \
#     --batch-size-valid 8 \
#     --max-learning-rate 1e-5 \
#     --gpus 0


# python train.py \
#     --stage LRFT \
#     --output-dir lrft_finetuned/20k_exp3 \
#     --model-path tpg_trained/model-06epoch-76566steps-1.4330loss \
#     --train-path data/lrft/kp20k_low-resource_20000_train_exp3.jsonl \
#     --valid-path data/lrft/kp20k_low-resource_valid.jsonl \
#     --batch-size-train 16 \
#     --batch-size-valid 8 \
#     --max-learning-rate 1e-5 \
#     --gpus 0
