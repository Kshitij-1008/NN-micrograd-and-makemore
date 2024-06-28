""" Using the Pre-Trained HuggingFace Transformer weights model and training my own.
Distributed Data Parallel (DDP) only initialization written. 
Datasets too big to be downloaded as well. DataLoader not edited due to same"""

#-----------------------------------------------------------------
# Imports
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time; import math
import inspect
import os
# ----------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1, """Flag as 1 for initialization"""
        # regularization
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # register buffer: Not needed after FlashAttention
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size() # (batch_size, sequence length, embdding dimensionality (n_embd))
        # calculate query, key, value for all heads in batch and move head forward
        # nh is "number of heads", hs is "head size", and C (number of channels) =nh*hs
        # eg. in GPT2 (124M): nh=12, hs=64, nh*hs=C = 768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        """
        att = q @ k.transpose(-1,-2) * k.size(-1)**-0.5 # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v #(B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        """
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True) #FlashAttention Implementation
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out 

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1, #For maintaining std of 1 for residual additions
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing --> efficiency
        self.transformer.wte.weight = self.lm_head.weight

        # initializating parameters
        self.apply(self._init_weights)
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02

            """Scaling the std for residual connections; multiplied by two because it's done at self.attn and self.mlp"""
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) **-0.5

            torch.nn.init.normal_(module.weight, std=std, mean=0.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None): 
        B,T = idx.shape
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is too small."
        
        # forward the positional and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # pos_emb shape: (T, n_embed)
        tok_emb = self.transformer.wte(idx) # tok_emb shape: (B,T, n_embed)
        x = tok_emb+pos_emb

        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        #forward the final layernorm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 Model weights from HuggingFace transformer"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("Loading weights from pre-trained GPT: %s" %model_type)

        # n_head, n_layer, n_embed are determined from model type
        config_args = {
            'gpt2' : dict(n_layer=12, n_head=12, n_embd=768),          # 124M Params
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024),  # 350M Params
            'gpt2-large' : dict(n_layer=36, n_head=20, n_embd=1280),   # 774M Params
            'gpt2-xl' : dict(n_layer=48, n_head=25, n_embd=1600),      # 1558M Params
        }[model_type]
        config_args['vocab_size'] = 50257 #256reg + 50000merges + 1 special token
        config_args['block_size'] = 1024 #vocab size and block size always same 

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args) 
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask

        # initialize a huggingface/transformer
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # remove unneeded buffers/filters/masks
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # some weights are transposed in tensorflow, so we have manually transpose them
        # whenever they are imported
        transposed = ['.attn.c_attn.weight', ".attn.c_proj.weight","mlp.c_fc.weight" ,"mlp.c_proj.weight"]
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {sd_keys}"
        for k in sd_keys_hf:
            if any (k.endswith(i) for i in transposed):
                # special treatment for the Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                # vanilla copy for others
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    # FusedAdamW & WeightDecay
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. 2D parameters will get weight decayed, i.e no layernorms or biases
        # only weights in linear+embeddings get selected.
        decay_params = [p for p in param_dict.values() if p.dim()>=2]
        nodecay_params = [p for p in param_dict.values() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed paramter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"Number of non-decayed paramter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if available (probably not)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ----------------------------------------------------------------

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
    
    # At initialization, load tokens and store in memory
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B*T)} batches")

        # state
        self.current_position = (self.B * self.T * self.process_rank) # 0 in our case
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += (B * T * self.num_processes) # (B * T) in our case
        # if loading the next batch will be out of bounds
        if self.current_position + (B * T * self.num_processes+1) > len(self.tokens):
            self.current_position = (self.B * self.T * self.process_rank) # 0 in our case
        return x, y

# ------------------------------------------------------------

"""Distributed Data Parallel (DDP) not done because I've only got 1 GPU lol"""


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 'WORLD_SIZE = no. of GPUs, 'RANK' = rank of GPU, 'LOCAL_RANK' = rank with nodes
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process does logging, checkpointing too
else:
    #establish the device
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

# pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

#  Reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# -----------------------------------------------------------

#Gradient Accumulation
total_batch_size = 32768 # 2**15, ~32k in number of tokens
B = 4
T = 256
assert total_batch_size % (B*T*ddp_world_size) == 0, f"Make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size//(B * T * ddp_world_size)
if master_process:
    print(f"Total batch tokens: {total_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

#get a data batch
train_loader = DataLoaderLite(B=4, T=256, process_rank=ddp_local_rank, num_processes=ddp_world_size)

# -----------------------------------------------------------

# for running matrix multiplication in Linear layers 8x faster, range is same, truncates mantissa
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------

# get the model
model = GPT(GPTConfig(vocab_size=50304)) #setting it to be nice number (a lot of powers of 2)
model.to(device)
using_compile = False
if using_compile:
    model = torch.compile(model) # Makes the code run faster
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# -----------------------------------------------------------

# Learning rate parameter tuning
"""Frist, there's a linear LR to its max value 
followed by Cosine decay upto 10% of its value
lastly, training continues at 10% of the original rate for the rest."""
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_steps iterations
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if iteration > lr_decay_iteration, lr=min_lr
    if it > max_steps:
        return min_lr
    # 3) Cosine decay if in between
    decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Invalid decay ratio"
    coeff = 0.5 * (1 + math.cos(decay_ratio*math.pi)) #coefficient starts at 1 and goes to 0
    return min_lr + (max_lr-min_lr) * coeff

# -----------------------------------------------------------

# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # hyperparameter tuning by setting betas and epsilon
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for iter in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # flushing the previous gradients
    loss_accum = 0.0

    #Gradient Accumulation to support larger batch size
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device); y = y.to(device)

        """In exchange of precision, shift to Bfloat16(BF16) with truncated mantissa but same range"""
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #In my case, this isn't that effective due to lack of powerful enough GPU
        logits, loss = model(x, y)
        loss /= grad_accum_steps # --> Normalizer! 
        loss_accum += loss.detach() #--> Otherwise, only the loss at the final microstep gets printed. ".detach()" separates the loss tensor from the graph
        
        # No need for synchronization of gradients for every microstep, inefficient.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)

        loss.backward()  # deposition of gradients

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping, Hyperparameter tuning

    # Determine and set learning rate for this iteration
    lr = get_lr(iter)
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr

    optimizer.step() # updation of parameters

    torch.cuda.synchronize() # waiting for the GPU to finish
    t1 = time.time()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)/(t1-t0)
    if master_process:
        print(f"Step {iter+1:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm.item():.4f} | dt: {(t1-t0)*1000:.6f} ms | tok/sec: {tokens_per_sec:.6f}")    

# -----------------------------------------------------------

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

num_return_sequences = 2
max_length = 30

# Getting the tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,") # 8 tokens [15496, 11, 314, 1101, 257, 3303, 2746, 11]
tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

"""Generating! x = (5,8) => B=5, T=8"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.shape[1]<max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits, _ = model.forward(x) #(B, T, vocab_size)

        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)

        # softmax probabilities 
        probs = F.softmax(logits, dim=-1)

        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs becomes (5, k) [here: (5,50)], topk_indices (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # sample a token from topk_probs
        ix = torch.multinomial(topk_probs, num_samples=1) #(B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        # append to original tensor
        x = torch.cat((x, xcol), dim=-1) # (5, 30) output


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {decoded}")

# -----------------------------------------------------------
