for sparsity in 0.0 0.5 0.75 0.875 0.9375
do
    for seed in 1 2 3 4 5
    do
    width=2048
    head_size=32
    n_heads=$((width / head_size))
    mkdir -p supar_examples/coord_check_shakespeare_char/sp/out
    out_dir="supar_examples/coord_check_shakespeare_char/sp/out/width${width}_depth2_sparsity${sparsity}_seed${seed}"
    python train.py \
        --out_dir=$out_dir \
        --eval_interval=1 \
        --log_interval=1 \
        --eval_iters=1 \
        --eval_only=False \
        --always_save_checkpoint=False \
        --never_save_checkpoint=True \
        --init_from='scratch' \
        --wandb_log=False \
        --csv_log=True \
        --dataset='shakespeare_char' \
        --gradient_accumulation_steps=4 \
        --batch_size=2 \
        --block_size=1024 \
        --n_layer=2 \
        --n_head=$n_heads \
        --n_embd=$width \
        --dropout=0.0 \
        --bias=False \
        --init_std=0.02 \
        --learning_rate=1e-2 \
        --max_iters=10 \
        --weight_decay=1e-1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --mup_enable_coord_check_logging=True \
        --sparsity=$sparsity \
        --seed=$seed \
        --backend='nccl' \
        --device='mps' \
        --dtype='float32' \
        --compile=False
    done
done
