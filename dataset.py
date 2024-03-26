import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class SongsDataset(Dataset):

    def __init__(self, ds, tokenizer_src, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) :
        src_traget_pair = self.ds[index]
        src_text = src_traget_pair['text']

        input_tokens = self.tokenizer_src.encode(src_text).ids

        num_padding_tokens = self.seq_len - len(input_tokens) - 1

        if num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "input": input,
            "mask": (input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "label": label,
            "src_text": src_text
        }



