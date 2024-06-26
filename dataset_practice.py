import torch
from torch.utils.data import Dataset
from typing import Any


class SongsDataset(Dataset):
    def __init__(self, data, tokenizer_src, seq_len):
        """
        Initialize the dataset.

        Parameters:
        - data: A list of pre-segmented and tokenized song lyrics as tensors.
        - seq_len: The sequence length for each segment.
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        """
        Returns a single data item at index `idx`.
        """
        segment = self.data[idx]
        segment = segment['text']


        # print(self.data)
        # # Calculate the start and end indices of the segment
        # start_idx = idx * self.seq_len
        # end_idx = start_idx + self.seq_len
        # # Extract the sequence starting at idx
        # input_segment = self.data[0][start_idx: end_idx]
        # # Extract the target sequence (offset by one token to predict the next token)
        # label_segment = self.data[0][start_idx + 1: idx + end_idx + 1]


        # Input is the entire segment except the last token
        input_segment = segment[:-1]
        # Label is the entire segment offset by one token (to predict the next token)
        label_segment = segment[1:]

        assert len(input_segment) == self.seq_len
        assert len(label_segment) == self.seq_len

        input_segment = torch.tensor(input_segment, dtype=torch.long)
        label_segment = torch.tensor(label_segment, dtype=torch.long)

        # Create a mask for the input segment
        # mask = (input_segment != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).int() & causal_mask(input_segment.size(0))

        return {
            "input": input_segment,
            # "mask": mask,
            "label": label_segment
        }

# def causal_mask(size):
#     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
#     return mask == 0