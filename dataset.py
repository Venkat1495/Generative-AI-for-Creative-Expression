import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class SongsDataset(Dataset):

    def __init__(self, ds, tokenizer_src, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.max_seq_len = seq_len -2 # account for [SOS] and [EOS]

        # self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        # self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        # self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        self.sos_token = tokenizer_src.token_to_id("[SOS]")
        self.eos_token = tokenizer_src.token_to_id('[EOS]')
        self.pad_token = tokenizer_src.token_to_id('[PAD]')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) :
        src_traget_pair = self.ds[index]
        src_text = src_traget_pair['text']

        input_tokens = self.tokenizer_src.encode(src_text).ids

        # Initialize segments
        input_segments = []
        label_segments = []
        masks = []

        # Generate segments and corresponding labels
        for i in range(0, len(input_tokens), self.max_seq_len):
            segment_tokens = input_tokens[i:i + self.max_seq_len]
            next_token = [input_tokens[i + self.max_seq_len]] if (i + self.max_seq_len) < len(input_tokens) else []
            segment = [self.sos_token] + segment_tokens + next_token + [self.eos_token]
            label = segment_tokens + next_token + [self.eos_token]  # Labels don't include [SOS] at the start

            num_padding = self.max_seq_len + 2 - len(segment) + 1  # Adjusting padding calculation
            segment += [self.pad_token] * num_padding
            label += [self.pad_token] * num_padding  # Ensure labels are padded similarly

            segment_tensor = torch.tensor(segment, dtype=torch.int64)
            label_tensor = torch.tensor(label, dtype=torch.int64)
            mask = [1] * (len(segment_tokens) + len(next_token) + 2) + [
                0] * num_padding  # Mask includes [SOS] and [EOS]
            mask_tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0).unsqueeze(0)

            input_segments.append(segment_tensor)
            label_segments.append(label_tensor)
            masks.append(mask_tensor)

        # Ensure input_segments and masks are tensors (possibly stacked)
        input_segments_tensor = torch.stack(input_segments)
        label_segments_tensor = torch.stack(label_segments)
        masks_tensor = torch.stack(masks)

        # num_padding_tokens = self.seq_len - len(input_tokens) - 1
        #
        # if num_padding_tokens < 0:
        #     raise ValueError('Sentence is too long')
        #
        # input = torch.cat(
        #     [
        #         self.sos_token,
        #         torch.tensor(input_tokens, dtype=torch.int64),
        #         torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
        #     ]
        # )
        #
        # label = torch.cat(
        #     [
        #         torch.tensor(input_tokens, dtype=torch.int64),
        #         self.eos_token,
        #         torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
        #     ]
        # )
        #
        # assert input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len

        return {
            "inputs": input_segments_tensor,
            "masks": masks_tensor,
            "labels": label_segments_tensor,
            "src_text": src_text
        }



