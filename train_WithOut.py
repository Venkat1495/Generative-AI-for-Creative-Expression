import warnings
import numpy as np
import torch
import os

from TransformerModel import build_transformer
from config import get_config, get_weights_file_path


import tiktoken
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter





def get_batch(split, train_data, val_data, device):
  # generate a small batch of bata of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - config['seq_len'], (config['seq_len'],))
  x = torch.stack([data[i:i+config['seq_len']] for i in ix])
  y = torch.stack([data[i+1:i+config['seq_len']+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out ={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split, train_data, val_data, device)
            logits, loss = model.decode(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




def get_model(config, vocab_src_len):
    # print(vocab_src_len)
    model = build_transformer(vocab_src_len, config['seq_len'], config['d_model'])
    return model



def train_model(config):
    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f'Using device {device}')
    tokenizer_src = tiktoken.get_encoding("gpt2")

    train_ds_size = np.memmap(os.path.join("Data", 'train.bin'), dtype=np.uint16, mode='r').astype(np.int64)
    val_ds_size = np.memmap(os.path.join("Data", 'val.bin'), dtype=np.uint16, mode='r').astype(np.int64)

    # Convert the numpy array to a PyTorch tensor
    train_data = torch.tensor(train_ds_size, dtype=torch.long)
    val_data = torch.tensor(val_ds_size, dtype=torch.long)

    # Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    model = get_model(config, vocab_src_len=50304).to(device)  # tokenizer_src.get_vocab_size()

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_weights_file_path(config) if preload == 'latest' else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        # initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])

        # global_step = state['global_step']
    else:
        print('No model to preload')

    # creating a Pytorch Optimizer
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    for step in range(62000):
        # every once in a while evaluate the loss on train and val sets
        if step % 15500 == 0:
            losses = estimate_loss(model, train_data, val_data, device)
            train_loss = losses['train']
            val_loss = losses['val']
            print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_folder = f"{config['model_folder']}"
                model_filename = f"{config['model_filename']}song.pt"
                model_filename = str(Path('.') / model_folder / model_filename)
                torch.save({
                    'iter_step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, model_filename)

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, device)

        # evaluate the loss
        logits, loss = model.decode(xb, yb)
        print(f"iter {step}: train loss {loss:.4f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    # starting_string = "The beginning of my story starts with"
    # Encode the starting string to token IDss
    print(f"Input: {config['generate_input']}")
    input = tokenizer_src.encode_ordinary(config['generate_input'])
    print(f"Input After Encode: {input}")
    input = (torch.tensor(input, dtype=torch.long, device=device)[None, ...])
    # input = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(f"Input After Encode and coverted to tensor: {input}")
    y = model.generate(config, input, 500)
    print(f"\n\nGenerated Output:\n{tokenizer_src.decode(y[0].tolist())}")
    # generate(device, tokenizer_src, model, config['generate_input'], lambda msg: batch_iterator.write(msg), 50)
    model.train()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)