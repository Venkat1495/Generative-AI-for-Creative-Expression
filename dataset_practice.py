import torch
from torch.utils.data import Dataset
import numpy as np

class SongsDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        Initialize the dataset.

        Parameters:
        - data_path: Path to the .bin file containing the pre-tokenized data.
        - seq_len: The sequence length for each training example.
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

        # Calculate number of complete sequences that can be extracted
        self.num_sequences = len(self.data) // (seq_len)

    def __len__(self):
        # Subtract seq_len to prevent indexing out of bounds when creating sequences
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns a single training example at index `idx`.

        Each training example is a tuple of (input_sequence, target_sequence) where:
        - input_sequence is the sequence of token IDs,
        - target_sequence is the sequence of token IDs offset by one position.
        """
        # Calculate the start of the sequence based on the index and sequence length
        start_index = idx * (self.seq_len)
        end_index = start_index + self.seq_len + 1  # +1 to include target sequence

        # Make sure we don't go out of bounds
        if end_index > len(self.data):
            raise IndexError("Index out of bounds")

        # Slice the data array to get the sequence for this index
        segment = self.data[start_index:end_index]

        input_segment = segment[:-1]  # All tokens except the last for input
        label_segment = segment[1:]  # All tokens except the first for labels

        return {
            "input": input_segment,
            "label": label_segment
        }



# import torch
# from torch.utils.data import Dataset
# from typing import Any
#
#
# class SongsDataset(Dataset):
#     def __init__(self, data, tokenizer_src, seq_len):
#         """
#         Initialize the dataset.
#
#         Parameters:
#         - data: A list of pre-segmented and tokenized song lyrics as tensors.
#         - seq_len: The sequence length for each segment.
#         """
#         super().__init__()
#         self.data = data
#         self.seq_len = seq_len
#         self.tokenizer_src = tokenizer_src
#
#     def __len__(self):
#         return len(self.data) - self.seq_len + 1
#
#     def __getitem__(self, idx):
#         """
#         Returns a single data item at index `idx`.
#         """
#         segment = self.data[idx]
#         segment = segment['text']
#
#
#         # Input is the entire segment except the last token
#         input_segment = segment[:-1]
#         # Label is the entire segment offset by one token (to predict the next token)
#         label_segment = segment[1:]
#
#         assert len(input_segment) == self.seq_len
#         assert len(label_segment) == self.seq_len
#
#         input_segment = torch.tensor(input_segment, dtype=torch.long)
#         label_segment = torch.tensor(label_segment, dtype=torch.long)
#
#         return {
#             "input": input_segment,
#             "label": label_segment
#         }
#
