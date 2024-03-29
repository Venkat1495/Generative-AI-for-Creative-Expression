import warnings

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Dataset as d
from dataset import SongsDataset, causal_mask
from TransformerModel import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


# def custom_collate_fn(batch):
#     # Determine the maximum number of segments and the maximum sequence length
#     max_segments = max(len(item['inputs']) for item in batch)
#     max_seq_length = max(max(seq.size(0) for seq in item['inputs']) for item in batch)
#
#     # Initialize containers for the batched data
#     batched_inputs = []
#     batched_masks = []
#     batched_labels = []
#     batched_texts = []
#
#     for item in batch:
#         # Pad the inputs and masks to the max_seq_length, and ensure each item has max_segments
#         inputs = [torch.nn.functional.pad(seq, (0, max_seq_length - seq.size(0)), 'constant', 0) for seq in item['inputs']]
#         masks = [torch.nn.functional.pad(mask, (0, 0, 0, max_seq_length - mask.size(3)), 'constant', 0) for mask in item['masks']]
#
#         # Pad the lists to have max_segments
#         inputs += [torch.zeros(max_seq_length) for _ in range(max_segments - len(inputs))]
#         masks += [torch.zeros(1, 1, max_seq_length) for _ in range(max_segments - len(masks))]
#
#         # Stack and append to the batch
#         batched_inputs.append(torch.stack(inputs))
#         batched_masks.append(torch.stack(masks))
#         batched_labels.append(item['label'])  # Adjust based on how you handle labels for multiple segments
#         batched_texts.append(item['src_text'])
#
#     # Convert lists to tensors
#     batched_inputs = torch.stack(batched_inputs)
#     batched_masks = torch.stack(batched_masks)
#     batched_labels = torch.stack(batched_labels)  # Adjust this line based on your actual label structure
#
#     return {'inputs': batched_inputs, 'masks': batched_masks, 'labels': batched_labels, 'src_text': batched_texts}



def greedy_decode(model, source, source_mask, tokenizer, max_len, device, print_msg):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    print_msg(str(source.shape))
    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # print_msg(str(decoder_input.shape))
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(decoder_input, decoder_mask, print_msg)

        # Get the next token
        prob = model.project(out[:, -1], print_msg)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    predicted = []


    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            inputs = batch['inputs'].to(device)
            masks = batch['masks'].to(device)

            for i in range(inputs.size(1)):
                input = inputs[:, i, :]
                mask = masks[:, i, :, :]
                # print_msg(str(mask.shape))
                model_out = greedy_decode(model, input, mask, tokenizer, max_len, device, print_msg)
                # print_msg(str(model_out))
                source_text = batch['src_text'][0]

                model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                predicted.append(model_out_text)

                # print to the console
                print_msg('-'*console_width)
                print_msg(f'SOURCE: {source_text}')
                print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                print_msg('-' * console_width)
                break






def get_all_sentences(ds):
    for item in ds['text']:
        yield item

def get_or_build_tokenizer(config, ds):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    ds_raw = pd.read_csv(config['path'])
    ds_raw = d.from_pandas(ds_raw)
    # print(type(ds_raw))

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw)

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = SongsDataset(train_ds_raw, tokenizer_src, config['seq_len'])
    val_ds = SongsDataset(val_ds_raw, tokenizer_src, config['seq_len'])

    max_len_src = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['text']).ids
        max_len_src = max(max_len_src, len(src_ids))

    # print(f'Max length of source sentence: {max_len_src}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloder = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloder, tokenizer_src

def get_model(config, vocab_src_len):
    # print(vocab_src_len)
    model = build_transformer(vocab_src_len, config['seq_len'], config['d_model'])
    return model

def train_model(config):

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloder, tokenizer_src = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        # print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(train_dataloader, desc=f'Precessing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()

            inputs = batch['inputs'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            batch_loss = 0
            for i in range(inputs.size(1)):
                input = inputs[:, i, :]
                mask = masks[:, i, :, :]
                label = labels[:, i, :]
                # batch_iterator.write(f"train loop mask Shape : {mask.shape}")
                decoder_output = model.decode(input, mask, lambda msg: batch_iterator.write(msg))
                proj_output = model.project(decoder_output, lambda msg: batch_iterator.write(msg))

                # batch_iterator.write(f"Projection Shape : {proj_output.shape}")
                # batch_iterator.write(f"Label Shape : {label.shape}")
                loss = loss_fn(proj_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))
                batch_loss += loss.item()

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            #Logging
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train.loss', loss.item(), global_step)
            writer.flush()
            global_step += 1

            run_validation(model, val_dataloder, tokenizer_src, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)






