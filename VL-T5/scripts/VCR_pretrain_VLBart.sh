# The name of experiment
name=VLBart

output=snap/vcr_pretrain/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/pretrain_vcr.py \
        --distributed --multiGPU --fp16 \
        --train train \
        --valid val \
        --batch_size 30 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --max_text_length 110 \
        --clip_grad_norm 1.0 \
        --losses 'lm,caption,refer,ground_caption' \
        --n_ground 4 \
        --backbone 'facebook/bart-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \

