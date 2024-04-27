import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken

# Hyperparamerters
path = 'data/part_5.csv'
batch_size = 12 # how many independent squences can be processed parallel?
block_size = 64 # what is the max contest lenght or predections ?
max_iters = 2000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.0
vocab_size = 50304
# ------------------

torch.manual_seed(1337)

# # Reading Data set
# df = pd.read_csv(path)
#
# # All the uniqie charecters that occurs in song lyrics
# text = df['lyrics'].str.cat(sep='\n')
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# # Creating the maping from charecter to intergers
# stoi = { ch:i for i,ch in enumerate(chars)}
# itos = { i:ch for i,ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s] # encoder : It takes string and output as list of integers
# decode = lambda l: ''.join([itos[i] for i in l])# decoder : It takes list on integers and outputs string
#
# # Now let split the data
# data = torch.tensor(encode(text), dtype = torch.long)
# n = int(0.9*len(data))
tokenizer_src = tiktoken.get_encoding("gpt2")
train_data = np.memmap(os.path.join("Data", 'train.bin'), dtype=np.uint16, mode='r').astype(np.int64) # data[:n]
val_data = np.memmap(os.path.join("Data", 'val.bin'), dtype=np.uint16, mode='r').astype(np.int64) # data[n:]

# Convert the numpy array to a PyTorch tensor
train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)

# data loading
def get_batch(split):
  # generate a small batch of bata of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out ={}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity """
    def  __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Tansformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# Super Simple biagram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # reducing the shape of the data to fit into the cross_entropy to check loss
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)  # calls forward funtions
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax from tje distribution
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel()
m = m.to(device)

# creating a Pytorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generator from the model
idx = f"Title: Watching Over Me; Tag: pop; Artist: Canadian Tenors;\n\n" + "Lyrics: \n"
input = tokenizer_src.encode_ordinary(idx)
# print(f"Input After Encode: {input}")
idx = (torch.tensor(input, dtype=torch.long, device=device)[None, ...])
# idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer_src.decode(m.generate(idx, max_new_tokens=500)[0].tolist()))


