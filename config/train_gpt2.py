# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 50

# eval stuff
eval_interval = 200
eval_iters = 100
log_interval = 10

moe_enabled      = True   # turn on with --moe_enabled=True
moe_num_experts  = 4       # experts per MoE layer
moe_top_k        = 1       # top‑k experts per token (1 or 2)
moe_router_type   = 'switch'   # 'switch', 'hash', or 'sinkhorn'
moe_sinkhorn_iters= 3
moe_aux_loss_coef = 0.01

# weight decay
weight_decay = 1e-1
