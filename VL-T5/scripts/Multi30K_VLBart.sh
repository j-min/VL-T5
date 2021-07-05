# The name of experiment
name=VLBart

output=snap/Multi30K/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/mmt.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test_2016_flickr,test_2017_flickr,test_2018_flickr \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 40 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \
        --num_beams 5 \
        --batch_size 80 \
        --max_text_length 40 \
        --gen_max_length 40 \
        --do_lower_case \