# The name of experiment
name=VLBart

output=snap/vcr/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vcr.py \
        --distributed --multiGPU --fp16 \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --output $output ${@:2} \
        --load snap/vcr_pretrain/VLBart/Epoch20 \
        --log_train_accuracy \
        --batch_size 12 \
        --valid_batch_size 36 \
        --max_text_length 110 \