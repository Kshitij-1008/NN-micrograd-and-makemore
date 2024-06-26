#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size=64
block_size=256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layers = 6
dropout = 0.2
# ------------------

torch.manual_seed(1337) #for reproducibility

with open(r"shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# All unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
#Encodings
encode = lambda s: [stoi[i] for i in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
n1 = int(0.9*len(data))
train_data = data[:n1]
val_data = data[n1:]

#Data loading
def get_batch(split):
    # select the appropriate dataset
    data = train_data if split == "train" else val_data
    
    #Generate random indices
    idx = torch.randint(0,len(data)-block_size, (batch_size,))
    
    #Select a block of text of size block_size starting from each random index
    x = torch.stack([data[i:i+block_size] for i in idx])
    
    #Shift the selected block of text by one character to derive the target set
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x = x.to(device); y= y.to(device)
    
    return x,y


@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model.forward(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


"""Head Module"""
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape    # C = no_embed
        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        #computing affinities (attention scores)
        wei = q @ k.transpose(-2,-1) * (self.head_size)**-0.5 # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) #(B,T,head_size)
        out = wei @ v #(B,T,head_size)

        return out
    

"""Multiple heads of self-attention in parallel"""
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h.forward(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


"""Feedforward Network"""
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


"""Transformer block: Communication followed by computation"""
class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        """ n_embed: embedding dimension
            n_head: number of heads in the MultiHeadAttention
        """
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # First LayerNorm applied before MultiHeadAttention
        self.ln1 = nn.LayerNorm(n_embed)
        # Second LayerNorm applied before FeedForward
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        # residual connection (adding the input to the output)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


"""BigramLanguage Model"""
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        """
        Before Block
        self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embed//4)
        self.ffwd = FeedForward(n_embed)
        """
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # shape of idx and targets: (B,T) tensor
        tok_emb = self.token_embedding_table(idx) #(B,T,C=n_embed) = (32,8,32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) = (8,32)
        x = pos_emb + tok_emb # (B,T,C) = (32,8,32); Broadcasting happens pos_emb(8,32)+tok_emb(32,8,32)
        """
        Before Block
        x = self.sa_heads(x) #Apply MultiHeadAttention after positional and token embeds 
        x = self.ffwd(x)  #Feedforward an MLP after establishing affinities
        """
        x = self.blocks(x) # shape: (B,T,n_embed) = (32,8,32)
        x = self.ln_f(x)  # shape: (B,T,n_embed=C) = (32,8,32)
        logits = self.lm_head(x) # shape: (B,T,vocab_size) = (32,8,65)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array tensor of indices in the current context
        for _ in range(max_new_tokens):
            # Positional embeds only go till block size, so have to regulate the context going in
            idx_cond = idx[:, -block_size:]
            #get the logits for the next token
            logits, loss = self.forward(idx_cond)
            #focusing on the only the last time step
            logits = logits[:, -1, :] #get the last time step for each sequence, shape: (B,C)
            #convert to probs using softmax
            probs = F.softmax(logits, dim=-1)
            #sample out an index
            idx_next = torch.multinomial(probs, num_samples=1)
            #append to existing index
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    #evaluate loss on train and val sets once a while
    if iter % eval_interval==0 or iter==max_iters-1:
        losses = estimate_losses()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch("train")
    
    #evaluate the loss
    logits, loss = m.forward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))