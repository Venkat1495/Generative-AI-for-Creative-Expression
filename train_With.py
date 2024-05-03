import warnings

import numpy as np
import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Dataset as d
# from dataset import SongsDataset, causal_mask
from dataset_practice import SongsDataset
from TransformerModel import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm




def validate_model(model, val_dataloader, device):
    total_val_loss = []

    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(val_dataloader):
            input = batch['input'].to(device)
            # mask = batch['mask'].to(device)
            label = batch['label'].to(device)

            output, loss = model.decode(input, label)
            # proj_output = model.project(decoder_output, print_msg)

            # loss = loss_fn(proj_output
            # .view(-1, tokenizer_src.get_vocab_size()), label.view(-1))
            if not torch.isnan(loss):
                total_val_loss.append(loss.item())
                # print(f"validation loss : {loss.item()}")

    average_val_loss = np.mean(total_val_loss)
    print(f"average validation loss : {average_val_loss}")
    return average_val_loss





def get_combin_segments(data, segment_size = 65):
    # Segment the data
    seg_len = len(data) // segment_size
    segmented_data = [data[i:i + segment_size] for i in range(seg_len)]
    # Convert the segmented data into a Hugging Face dataset
    dataset = d.from_dict({'text': segmented_data})
    return dataset




def get_ds(config):

    tokenizer_src = tiktoken.get_encoding("gpt2")

    path = config["path"]

    # train_ds_size = np.memmap(os.path.join("Data", 'train.bin'), dtype=np.uint16, mode='r')    #.astype(np.int64)
    # val_ds_size = np.memmap(os.path.join("Data", 'val.bin'), dtype=np.uint16, mode='r')    #.astype(np.int64)

    # train_ds_size = get_combin_segments(train_ds_size)
    # val_ds_size = get_combin_segments(val_ds_size)

    # train_ds_size, val_ds_size = random_split(ds_raw, [train_size, val_size])

    train_ds_size = np.memmap(os.path.join(path, 'train.bin'), dtype=np.uint16, mode='r').astype(np.int64)
    val_ds_size = np.memmap(os.path.join(path, 'val.bin'), dtype=np.uint16, mode='r').astype(np.int64)

    # Convert the numpy array to a PyTorch tensor
    train_data = torch.tensor(train_ds_size, dtype=torch.long)
    val_data = torch.tensor(val_ds_size, dtype=torch.long)

    train_ds = SongsDataset(train_data, config['seq_len'])
    val_ds = SongsDataset(val_data, config['seq_len'])

    train_data = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_data = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_data, val_data, tokenizer_src

def get_model(config, vocab_src_len):
    # print(vocab_src_len)
    model = build_transformer(vocab_src_len, config['seq_len'], config['d_model'])
    return model



def cheking_point(model, val_dataloder, epoch, global_step, optimizer, tokenizer_src, device):
    print("checking")
    model.eval()
    print("checking1")
    # average_val_loss = validate_model(model, val_dataloder, device)
    average_val_loss = 2
    print("checking2")
    model.train()
    print("checking3")
    config = get_config()
    print("checking4")
    # best_val_loss = average_val_loss  # Update the best validation loss
    model_folder = f"{config['model_folder']}"
    print("checking5")
    model_filename = f"{config['model_filename']}_{epoch}.pt"
    print("checking6")
    model_filename = str(Path('.') / model_folder / model_filename)
    print("checking7")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)
    print("checking8")

    model.eval()
    print("working")
    # Encode the starting string to token IDss
    print(f"Input: {config['generate_input']}")
    input = tokenizer_src.encode_ordinary(config['generate_input'])
    input = (torch.tensor(input, dtype=torch.long, device=device)[None, ...])
    print(f"Input After Encode and coverted to tensor: {input}")
    y = model.generate(config, input, 500, tokenizer_src)
    print(f"\n\nGenerated Output:\n{tokenizer_src.decode(y[0].tolist())}")
    model.train()
    return average_val_loss

def train_model(config):

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloder, tokenizer_src = get_ds(config)
    print(len(train_dataloader))
    print(len(train_dataloader)//config['seq_len'])
    model = get_model(config, vocab_src_len=50304).to(device) # tokenizer_src.get_vocab_size()

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_weights_file_path(config) if preload == 'latest' else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=torch.device(device))
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload')

    quarter_point = len(train_dataloader) // 4  # Calculate the 25% point
    average_val_loss = 0
    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Precessing epoch {epoch:02d}')
        total_loss = []  # For accumulating loss over the entire epoch
        # total_segments = 0

        for j, batch in enumerate(batch_iterator):
            model.train()

            input = batch['input'].to(device)
            label = batch['label'].to(device)

            output, loss = model.decode(input, label)


            if torch.isnan(loss):
                print(f"Loss is NaN. Skipping...batch number : {j}")
                continue
            total_loss.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()




            if  j % 2000 == 0: # j % 2000 == 0 and

                average_val_loss = cheking_point(model, val_dataloder, epoch, global_step, optimizer, tokenizer_src, device)

            #Logging

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) # , "val_loss": f"{average_val_loss:.4f}"})
            writer.add_scalar('train.loss', loss.item(), global_step)
            writer.add_scalar('val.loss', average_val_loss, global_step)
            writer.flush()
            global_step += 1

        average_val_loss = cheking_point(model, val_dataloder, epoch, global_step, optimizer, tokenizer_src, device)

        print(f"End of Epoch: {epoch}. Final Validation Loss : {average_val_loss}. Final training loss: {np.mean(total_loss)}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)