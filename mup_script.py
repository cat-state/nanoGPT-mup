# %% [markdown]
# ### Using MuP to Tune Hyperparameters
# 
# This is the accompanying notebook for (TODO LINK TO BLOG). In this notebook, we generate all of the figures used in the blog.
# 
# We wish to transfer from xxx to yyy

# %%
# Import local nanoGPT
from model import GPTConfig, GPT
import torch

# %%
# -----------------------------------------------------------------------------
# default config values from train.py designed to train on OpenWebText.

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'mup'
wandb_run_name = 'transfer' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# muP settings
mup_width_mult = 1.0

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster


# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# %%
def params():
    """ estimates the number of parameters in the model"""
    out = OrderedDict()

    # token and position embeddings
    out['emebedding/position'] = n_embd * block_size
    out['embedding/token'] = n_embd * vocab_size
    out['embedding'] = out['emebedding/position'] + out['embedding/token']

    # attention blocks
    out['attention/ln'] = n_embd # note, bias=False in our LN
    out['attention/kqv'] = n_embd * 3*n_embd
    out['attention/proj'] = n_embd**2
    out['attention'] = out['attention/ln'] + out['attention/kqv'] + out['attention/proj']

    # MLP blocks
    ffw_size = 4*n_embd # feed forward size
    out['mlp/ln'] = n_embd
    out['mlp/ffw'] = n_embd * ffw_size
    out['mlp/proj'] = ffw_size * n_embd
    out['mlp'] = out['mlp/ln'] + out['mlp/ffw'] + out['mlp/proj']
    
    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['ln_f'] = n_embd # final layernorm
    out['dense'] = 0 # 0 because of parameter sharing. This layer uses the weights from the embedding layer

    # total
    out['total'] = out['embedding'] + out['transformer'] + out['ln_f'] + out['dense']

    return out

# compare our param count to that reported by PyTorch
p = params()
params_total = p['total']
print(f"we see: {params_total}, expected: {124337664}, match: {params_total == 124337664}")
# create a header
print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
for k,v in p.items():
    print(f"{k:20s} {v:10d} {v/params_total*100:10.4f}")
    

# %%
# we can now calculate the size of each checkpoint
# params are stored in fp32, and the AdamW optimizer has 2 additional buffers per param for statistics
params_bytes = params_total*4
params_and_buffers_bytes = params_bytes + 2*params_bytes
print(f"est checkpoint size: {params_and_buffers_bytes/1e9:.2f} GB")
measured_bytes = 1542470366 # from wc -c ckpt.pt
print(f"measured with wc -c ckpt.pt: {measured_bytes}")
print(f"fluff ratio: {measured_bytes/params_and_buffers_bytes*100:.2f}%")

# %% [markdown]
# We can also estimate the ratio of our GPU memory that will be taken up just by the weights and the buffers inside the AdamW optimizer

# %%
gpu_memory = 40e9 # 40 GB A100 GPU, roughly
print(f"memory ratio taken up just for parameters: {params_and_buffers_bytes / gpu_memory * 100:.2f}%")

# %% [markdown]
# i.e. not that much of the memory for this tiny model, most of the memory is activations (forward and backward). This of course changes dramatically for larger and larger models.

# %% [markdown]
# Let's estimate FLOPs for a single forward pass.

# %%
def flops():
    # we only count Weight FLOPs, all other layers (LayerNorm, Softmax, etc) are effectively irrelevant
    # we count actual FLOPs, not MACs. Hence 2* all over the place
    # basically for any matrix multiply A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D

    out = OrderedDict()
    head_size = n_embd // n_head

    # attention blocks
    # 1) the projection to key, query, values
    out['attention/kqv'] = 2 * block_size * (n_embd * 3*n_embd)
    # 2) calculating the attention scores
    out['attention/scores'] = 2 * block_size * block_size * n_embd
    # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out['attention/reduce'] = 2 * n_head * (block_size * block_size * head_size)
    # 4) the final linear projection
    out['attention/proj'] = 2 * block_size * (n_embd * n_embd)
    out['attention'] = sum(out['attention/'+k] for k in ['kqv', 'scores', 'reduce', 'proj'])

    # MLP blocks
    ffw_size = 4*n_embd # feed forward size
    out['mlp/ffw1'] = 2 * block_size * (n_embd * ffw_size)
    out['mlp/ffw2'] = 2 * block_size * (ffw_size * n_embd)
    out['mlp'] = out['mlp/ffw1'] + out['mlp/ffw2']

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['dense'] = 2 * block_size * (n_embd * vocab_size)

    # forward,backward,total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total'] # use common estimate of bwd = 2*fwd
    out['total'] = out['forward_total'] + out['backward_total']

    return out
    
# compare our param count to that reported by PyTorch
f = flops()
flops_total = f['forward_total']
print(f"{'name':20s} {'flops':14s} {'ratio (%)':10s}")
for k,v in f.items():
    print(f"{k:20s} {v:14d} {v/flops_total*100:10.4f}")
    

# %%
# now here is an estimate copy pasted from the PaLM paper
# this formula is often used to calculate MFU (model flops utilization)
def palm_flops():
    """estimate of the model flops following PaLM paper formula"""
    # non-embedding model parameters. note that we do not subtract the
    # embedding/token params because those are tied and get used in the last layer.
    N = params()['total'] - params()['emebedding/position']
    L, H, Q, T = n_layer, n_head, n_embd//n_head, block_size
    mf_per_token = 6*N + 12*L*H*Q*T
    mf = mf_per_token * block_size
    return mf

print(f"palm_flops: {palm_flops():d}, flops: {flops()['total']:d}, ratio: {palm_flops()/flops()['total']:.4f}")

# %% [markdown]
# Ok they are quite similar, giving some confidence that my math in flops() function was ~ok. Now, A100 is cited at 312TFLOPS bfloat16 on tensor cores. So what is our model flops utilization (MFU)? I trained the model above with a batch_size of 20 and grad_accum of 5, which runs in about 755ms on a single A100 GPU. We get:

# %%
# here is what we currently roughly measure
batch_size = 20 * 5 # 5 is grad_accum, so total batch size is 100
measured_time = 0.755 # in seconds per iteration
measured_throughput = batch_size / measured_time
flops_achieved = f['total'] * measured_throughput

# A100 is cited to be 312 TFLOPS of bloat16 running on tensor cores
a100_flops_promised = 312e12

# the fraction of the A100 that we are using:
print(f"fraction of A100 used: {flops_achieved / a100_flops_promised * 100:.2f}%")

# %% [markdown]
# For reference, we'd prefer to be somewhere around 50%+, and not just for a single GPU but for an entire DDP run. So we still have some work to do, but at least we're within a factor of ~2X of what is achievable with this GPU.

# %%
# Finally let's check out the 6ND approximation as total cost of training in FLOPs
model_size = params()['total'] # this is number of parameters, N
tokens_num = 300e9 # 300B tokens, this is dataset size in tokens, D
a100_flops = 312e12 # 312 TFLOPS
assumed_mfu = 0.3 # assume this model flops utilization (take the current 37% from above and add some DDP overhead)
flops_throughput = a100_flops * 8 * assumed_mfu # assume an 8XA100 node at 30% utilization
flops_needed = 6 * model_size * tokens_num # 6ND
time_needed_s = flops_needed / flops_throughput # in seconds
print(f"time needed to train the model: {time_needed_s/3600/24:.2f} days")

# %% [markdown]
# This is not a bad estimate at all. I trained this model and it converged in roughly 4 days. Btw as a good reference for where 6ND comes from and some intuition around it I recommend [Dzmitry's post](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).

# %% [markdown]
# Now, FLOPs are just one constraint, the other that we have to keep a close track of is the memory bandwidth. TODO estimate LOAD/STORE costs of our model later.


