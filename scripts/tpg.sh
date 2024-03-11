mkdir tpg_trained

python train.py \
    --stage TPG \
    --output-dir tpg_trained \
    --model-path facebook/bart-base \
    --train-path data/tpg/kp20k_TPG_train.jsonl \
    --valid-path data/tpg/kp20k_TPG_valid.jsonl \
    --batch-size-train 32 \
    --batch-size-valid 16 \
    --max-learning-rate 2e-4 \
    --gpus 0 