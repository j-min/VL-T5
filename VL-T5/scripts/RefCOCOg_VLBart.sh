# The name of experiment
name=VLBart

output=snap/refcocog/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/refcoco.py \
        --distributed --multiGPU \
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
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \
        --batch_size 90 \
