import warnings

import numpy as np
import torch
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


#
# def greedy_decode(model, source, source_mask, tokenizer, max_len, device, print_msg):
#     sos_idx = tokenizer.token_to_id('[SOS]')
#     eos_idx = tokenizer.token_to_id('[EOS]')
#     print_msg(str(source.shape))
#     # initialize the decoder input with the sos token
#     decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
#     while True:
#         if decoder_input.size(1) == max_len:
#             break
#         # print_msg(str(decoder_input.shape))
#         decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
#
#         out = model.decode(decoder_input, decoder_mask, print_msg)
#
#         # Get the next token
#         prob = model.project(out[:, -1], print_msg)
#         _, next_word = torch.max(prob, dim=1)
#         decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
#
#         if next_word == eos_idx:
#             break
#
#     return decoder_input.squeeze(0)
#
#
#
# def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):
#     model.eval()
#     count = 0
#
#     source_texts = []
#     predicted = []
#
#
#     console_width = 80
#
#     with torch.no_grad():
#         for batch in validation_ds:
#             count += 1
#             inputs = batch['inputs'].to(device)
#             masks = batch['masks'].to(device)
#
#             for i in range(inputs.size(1)):
#                 input = inputs[:, i, :]
#                 mask = masks[:, i, :, :]
#                 # print_msg(str(mask.shape))
#                 model_out = greedy_decode(model, input, mask, tokenizer, max_len, device, print_msg)
#                 # print_msg(str(model_out))
#                 source_text = batch['src_text'][0]
#
#                 model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())
#
#                 source_texts.append(source_text)
#                 predicted.append(model_out_text)
#
#                 # print to the console
#                 print_msg('-'*console_width)
#                 print_msg(f'SOURCE: {source_text}')
#                 print_msg(f'PREDICTED: {model_out_text}')
#
#             if count == num_examples:
#                 print_msg('-' * console_width)
#                 break

def generate(device, tokenizer, model, input, print_msg, max_length: int = 100):
    # Define the device and load configurations
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # print("Using device:", device)
    # config = get_config()
    #
    # # Initialize the tokenizer and model
    # tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    # model = build_transformer(tokenizer.get_vocab_size(), config['seq_len'], config['d_model']).to(device)
    #
    # # Load the pretrained weights
    # model_filename = get_weights_file_path(config)  # Update function name as needed
    # state = torch.load(model_filename)
    # state_dic = state['model_state_dict']
    # model.load_state_dict(state_dic)

    # model.eval()
    with torch.no_grad():
        # Encode the seed text
        encoded_input = tokenizer.encode(input).ids
        input_ids = torch.tensor([tokenizer.token_to_id('[SOS]')] + encoded_input, dtype=torch.int64).unsqueeze(0).to(
            device)

        # Generation loop
        for _ in range(max_length - len(encoded_input)):

            # mask = (input_ids != tokenizer.token_to_id('[PAD]')).unsqueeze(0).int() & causal_mask(input_ids.size(1), device) # Update or create a mask if your model requires it
            mask = torch.tril(torch.ones((1, input_ids.size(1), input_ids.size(1))), diagonal=1).type(torch.int).to(
                device)

            output = model.decode(input_ids, mask, print_msg)  # Adjust this call based on your model's method signature
            proj_output = model.project(output[:, -1, :], print_msg)  # Adjust if necessary to match your model's output format
            # idx_next = torch.multinomial(proj_output, num_samples=1)
            # print(tokenizer.decode(proj_output[0].tolist()))
            # # Get the next token (you might want to sample instead of using argmax for more diversity)
            _, idx_next = torch.max(proj_output, dim=1)
            input_ids = torch.cat([input_ids, torch.tensor([[idx_next]], dtype=torch.int64).to(device)], dim=1)

            # print the translated word
            print(f"{tokenizer.decode([idx_next])}", end=' ')
            # Break if [EOS]
            if idx_next == tokenizer.token_to_id('[EOS]'):
                break

        # Decode the generated IDs to text
        generated_text = tokenizer.decode(input_ids[0].tolist())
        print(f"\nGenerated song:\n{generated_text}")


def validate_model(model, val_dataloader, device):
    total_val_loss = []

    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(val_dataloader):
            input = batch['input'].to(device)
            # mask = batch['mask'].to(device)
            label = batch['label'].to(device)

            output, loss = model.decode(input, label)
            # proj_output = model.project(decoder_output, print_msg)

            # loss = loss_fn(proj_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))
            if not torch.isnan(loss):
                total_val_loss.append(loss.item())
                # print(f"validation loss : {loss.item()}")

    average_val_loss = np.mean(total_val_loss)
    print(f"average validation loss : {average_val_loss}")
    return average_val_loss





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

def get_segments(config, input, num_segments):
    # Add padding to ensure the length meets a multiple of max_seq_len
    if len(input) % config['segment_len'] == 0:
        # print(len(input))
        # # Split into segments of length max_seq_len
        # num_segments = len(input) // config['segment_len']
        input_segments = []

        for i in range(num_segments):
            start_idx = i * config['segment_len']
            end_idx = start_idx + config['segment_len']
            input_segment = input[start_idx:end_idx]
            input_segments.append(input_segment)
    else:
        print("Padding is not matching, please check")
    return input_segments

def get_combin_segments(data, segment_size = 301):
    # Segment the data
    seg_len = len(data) // segment_size
    segmented_data = [data[i:i + segment_size] for i in range(seg_len)]
    # Convert the segmented data into a Hugging Face dataset
    dataset = d.from_dict({'text': segmented_data})
    input_length = []
    for item in dataset['text']:
        input_length.append(len(item))
    count_less_than_100 = len([x for x in input_length if x < 301])
    count_100_to_200 = len([x for x in input_length if 302 <= x])
    print(count_less_than_100)
    print(count_100_to_200)
    return dataset


def get_data(config, tokenizer_src, ds_raw):
    sos_token = tokenizer_src.token_to_id('[SOS]')
    eos_token = tokenizer_src.token_to_id('[EOS]')
    pad_token = tokenizer_src.token_to_id('[PAD]')
    inputs = []

    for item in ds_raw['text']:
        input_tokens = tokenizer_src.encode_ordinary(item)
        input_length = len(input_tokens)

        # Adjust config['max_seq_len'] based on input tokens length
        if input_length <= 299:
            config['max_seq_len'] = 301
            num_segments = 1
        elif 299 < input_length <= 600:
            config['max_seq_len'] = 602
            num_segments = 2
        elif 600 < input_length <= 901:
            config['max_seq_len'] = 903
            num_segments = 3
        elif 901 < input_length <= 1202:  # For input_length > 901 and up to 1204
            config['max_seq_len'] = 1204
            num_segments = 4
        # print(len(input_tokens))
        dec_num_padding_tokens = config['max_seq_len'] - len(input_tokens) - 2


        input = [sos_token] + input_tokens +  [eos_token] + [pad_token] * dec_num_padding_tokens
        # print(len(input))
        inputs.extend(get_segments(config, input, num_segments))

    return d.from_dict({"text": inputs})


def get_combine_data(ds_raw, tokenizer, sos_token="<|startoftext|>", eos_token="<|endoftext|>"):

    # input = []

    # Modify each item in the 'text' column to include SOS and EOS tokens
    ds_raw['text'] = ds_raw['text'].apply(lambda x: sos_token + x + eos_token)

    data = ds_raw['text'].str.cat(sep='\n')

    data = tokenizer.encode_ordinary(data)


    return data

def get_ds(config):
    # ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    ds_raw = pd.read_csv(config['path'])
    # ds_raw = d.from_pandas(ds_raw)

    # Build tokenizers
    # tokenizer_src = get_or_build_tokenizer(config, ds_raw)
    tokenizer_src = tiktoken.get_encoding("gpt2")

    # ds_raw = get_data(config, tokenizer_src, ds_raw)

    ds_raw = get_combine_data(ds_raw, tokenizer_src)
    ds_raw = get_combin_segments(ds_raw)


    # # keep 90% for training and 10% for validation
    # # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_size, val_ds_size = random_split(ds_raw, [train_ds_size, val_ds_size])

    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Assuming ds_raw is a Dataset object
    total_size = len(ds_raw)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # Split the dataset
    train_ds_size, val_ds_size = random_split(ds_raw, [train_size, val_size])

    train_ds = SongsDataset(train_ds_size, tokenizer_src, config['seq_len'])
    val_ds = SongsDataset(val_ds_size, tokenizer_src, config['seq_len'])

    # max_len_src = 0

    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item['text']).ids
    #     max_len_src = max(max_len_src, len(src_ids))

    # print(f'Max length of source sentence: {max_len_src}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    val_dataloder = DataLoader(val_ds, batch_size=1, shuffle=False)

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
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload')

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
            # mask = batch['mask'].to(device)
            label = batch['label'].to(device)

            output, loss = model.decode(input, label)
            # proj_output = model.project(decoder_output, lambda msg: batch_iterator.write(msg))

            # loss = nn.CrossEntropyLoss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                # loss_fn(proj_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))

            if torch.isnan(loss):
                print(f"Loss is NaN. Skipping...batch number : {j}")
                continue
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)



            # losses = torch.zeros(inputs.size(1))
            # for i in range(inputs.size(1)):
            #     input = inputs[:, i, :]
            #     mask = masks[:, i, :, :]
            #     label = labels[:, i, :]
            #     # batch_iterator.write(f"train loop mask Shape : {mask.shape}")
            #     decoder_output = model.decode(input, mask, lambda msg: batch_iterator.write(msg))
            #     proj_output = model.project(decoder_output, lambda msg: batch_iterator.write(msg))
            #
            #     # batch_iterator.write(f"Projection Shape : {proj_output.shape}")
            #     # batch_iterator.write(f"Label Shape : {label.shape}")
            #     loss = loss_fn(proj_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))
            #     loss.backward()
            #     losses[i] += loss.item()
            #     total_segments += 1

            # segment_loss = losses.mean()  # Accumulate loss over the epoch
            # # Average or sum the loss across segments here if desired
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)


            average_val_loss = 0
            # if j % 500 == 0 and j != 0:
            #     model.eval()
            #     average_val_loss = validate_model(model, val_dataloder, loss_fn, device, lambda msg: batch_iterator.write(msg), tokenizer_src)
            #     model.train()
            #
            # if j % 500 == 0 and average_val_loss < best_val_loss and j != 0:
            #     print(f"Validation loss decreased ({best_val_loss:.4f} --> {average_val_loss:.4f}). Saving model...")
            #     best_val_loss = average_val_loss  # Update the best validation loss
            #     model_folder = f"{config['model_folder']}"
            #     model_filename = f"{config['model_filename']}{epoch}_{global_step}.pt"
            #     model_filename = str(Path('.') / model_folder / model_filename)
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'global_step': global_step
            #     }, model_filename)

            #Logging

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}" , "val_loss": f"{average_val_loss:.4f}"})
            writer.add_scalar('train.loss', loss.item(), global_step)
            writer.add_scalar('val.loss', average_val_loss, global_step)
            writer.flush()
            global_step += 1

        model.eval()
        average_val_loss = validate_model(model, val_dataloder, device)
                                              # lambda msg: batch_iterator.write(msg), tokenizer_src)
        model.train()
        print(f"End of Epoch: {epoch}. Final Validation Loss : {average_val_loss}. Final training loss: {np.mean(total_loss)}")

        if average_val_loss < best_val_loss :
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {average_val_loss:.4f}). Saving model...")
            best_val_loss = average_val_loss  # Update the best validation loss
            model_folder = f"{config['model_folder']}"
            model_filename = f"{config['model_filename']}{epoch}_{global_step}.pt"
            model_filename = str(Path('.') / model_folder / model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

        model.eval()
        y = model.generate(config['generate_input'],50)
        print(tokenizer_src.decode(y[0].tolist()))
        # generate(device, tokenizer_src, model, config['generate_input'], lambda msg: batch_iterator.write(msg), 50)
        model.train()
        # run_validation(model, val_dataloder, tokenizer_src, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        # Example usage
        # After processing all batches


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)