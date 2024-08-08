import os
import time
import math
import pickle
from contextlib import nullcontext
import wandb
import gc

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Configuration
def get_config():
    config = {
        # I/O
        'out_dir': 'out',
        'eval_interval': 2000,
        'log_interval': 400,
        'eval_iters': 200,
        'eval_only': False,
        'always_save_checkpoint': True,
        'init_from': 'scratch',
        
        # Wandb logging
        'wandb_log': False,
        'wandb_project': 'owt',
        'wandb_run_name': 'gpt2',
        
        # Data
        'dataset': 'openwebtext',
        'gradient_accumulation_steps': 2,
        'batch_size': 2,
        'block_size': 1024,
        
        # Model
        'n_layer': 12,
        'n_head': 16,
        'n_embd': 512,
        'dropout': 0.0,
        'bias': False,
        
        # Optimizer
        'learning_rate': 6e-4,
        'max_iters': 400,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Learning rate decay
        'decay_lr': False,
        'warmup_iters': 2000,
        'lr_decay_iters': 600000,
        'min_lr': 6e-5,
        'start_iter_num': 0,
        'start_best_val_loss': 1e9,
        
        # muP settings
        'use_mup': True,
        'mup_width_mult': 2048. / 64.,
        'mup_width_list': [2048],# [64, 128, 256, 512, 1024, 2048],
        'mup_lr_list': [2**(-8), 2**(-7), 2**(-6),], #[2**(-20), 2**(-18), 2**(-16), 2**(-14), 2**(-12), 2**(-10), 2**(-8), 2**(-6)],
        
        # DDP settings
        'backend': 'nccl',
        
        # System
        'device': 'cuda',
        'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
        'compile': True,
    }
    return config

# Setup
def setup_training(config):
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    # Set up directories and seeds
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    config['device_type'] = device_type
    
    return ddp, master_process, seed_offset, device_type, ctx

# Data loading
def get_batch(split, config):
    data_dir = os.path.join('data', config['dataset'])
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    if config['device_type'] == 'cuda':
        x, y = x.pin_memory().to(config['device'], non_blocking=True), y.pin_memory().to(config['device'], non_blocking=True)
    else:
        x, y = x.to(config['device']), y.to(config['device'])
    return x, y

# Model initialization
def init_model(config, meta_vocab_size=None):
    model_args = dict(n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'],
                      block_size=config['block_size'], bias=config['bias'], vocab_size=None,
                      dropout=config['dropout'], use_mup=config['use_mup'], mup_width_mult=config['mup_width_mult'])
    
    if config['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=config['device'])
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        config['start_iter_num'] = checkpoint['iter_num']
        config['start_best_val_loss'] = checkpoint['best_val_loss']
    elif config['init_from'].startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']
    
    return model, model_args

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split, config)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(config, it):
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

# Training loop
def train(model, config, ddp, master_process, ctx, get_batch):

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))

    # Optimizer
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'],
                                           (config['beta1'], config['beta2']), config['device_type'])
    
    if config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Compile the model
    if config['compile']:
        print("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    
    # Wrap model into DDP container
    if ddp:
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Training loop
    X, Y = get_batch('train', config)
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    iter_num = config['start_iter_num']
    best_val_loss = config['start_best_val_loss']
    loss_avg = 1

    
    while True:
        # Learning rate decay
        lr = get_lr(config, iter_num) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #for pn, p in model.named_parameters():
        #    print(pn)
        # Create a dictionary to map parameter names to optimizer param groups
        param_to_group = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                param_to_group[p] = group

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param in param_to_group:
                    param_group = param_to_group[param]

                    # Check if the parameter is part of attention or MLP layers
                    if any(proj in name for proj in ['c_attn', 'c_fc', 'c_proj']) and name.endswith(('weight', 'bias')):
                        param_group['lr'] = lr / config['mup_width_mult']
                        #print(f'mup param: {name}')
                    else:
                        param_group['lr'] = lr
                        #print(f'NOT mup param: {name}')
                else:
                    print(f"Warning: Parameter {name} not found in optimizer. Skipping.")
        
        # Evaluate the model
        if iter_num % config['eval_interval'] == 0 and master_process:
            losses = estimate_loss(config)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if config['wandb_log']:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                })
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
        if iter_num == 0 and config['eval_only']:
            break
        
        # Forward and backward pass
        for micro_step in range(config['gradient_accumulation_steps']):
            if ddp:
                model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config['gradient_accumulation_steps']
            X, Y = get_batch('train', config)
            scaler.scale(loss).backward()
        
        # Gradient clipping and optimizer step
        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if local_iter_num == 5:
            loss_avg = loss.item() * config['gradient_accumulation_steps']
        if local_iter_num > 5:
            loss_avg -= loss_avg / iter_num
            loss_avg += loss.item() * config['gradient_accumulation_steps']  / iter_num
        if iter_num % config['log_interval'] == 0 and master_process:
            lossf = loss.item() * config['gradient_accumulation_steps']
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, avg_loss {loss_avg:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1
        
        # Check for termination
        if iter_num > config['max_iters']:
            break

    if ddp:
        destroy_process_group()

    return math.log2(loss_avg)

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Setup training environment
    ddp, master_process, seed_offset, device_type, ctx = setup_training(config)
    
    avg_losses = {}
    for width in config['mup_width_list']:
        for lr in config['mup_lr_list']:
            print(f'lr: {math.log2(lr)}')
            print(f'width: {width}')

            config['n_embd'] = width
            config['learning_rate'] = lr

            # Initialize model
            model, model_args = init_model(config)

            model.to(device_type)

            # Start training
            avg_losses[(width, lr)] = train(model, config, ddp, master_process, ctx, get_batch)

            del model
            gc.collect()
            torch.cuda.empty_cache()
    print(avg_losses)
