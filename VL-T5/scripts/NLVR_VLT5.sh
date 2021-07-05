# The name of experiment
name=VLT5

output=snap/nlvr/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/nlvr.py \
        --distributed --multiGPU \
        --train train \
        --valid valid \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLT5/Epoch30 \
        --batch_size 30 \
        --max_text_length 40 \