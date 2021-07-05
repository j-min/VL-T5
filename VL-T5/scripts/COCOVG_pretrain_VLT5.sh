# The name of experiment
name=VLT5

output=snap/pretrain/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 80 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'lm,qa,ground_caption,refer,itm' \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --epoch 30 \


