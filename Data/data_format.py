import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from tiktoken import get_encoding

# Load your dataset
df = pd.read_csv('part_3.csv')


# Function to format and preprocess text
def preprocess_text(text):
    # Clean lyrics by removing unwanted characters and truncating if too long
    # text = text.replace('[', '').replace(']', '')
    sos_token = "[SOS]"  # Define your SOS token
    eos_token = "[EOS]"  # Define your EOS token
    return f"{sos_token} {text} {eos_token}"


# Define the 'text' column by concatenating question and answer
df['text'] = df.apply(lambda row: f"Title: {row['title']}; Tag: {row['tag']}; Artist: {row['artist']};\n\n"
                                  f"Lyrics : \n{preprocess_text(row['lyrics'])}\n\n\n\n", axis=1)

print(df['text'][0])

# Create a train-test split
train_df = df.sample(frac=0.98, random_state=123)  # 90% training data
val_df = df.drop(train_df.index)  # Remaining 10% validation data

# Initialize the tokenizer
encoder = get_encoding("gpt2")  # Use your desired model


# Tokenize and save function
def tokenize_and_save(df, filename):
    # Calculate the length of the dataset after tokenization
    token_lengths = df['text'].apply(lambda text: len(encoder.encode_ordinary(text)))
    if filename == "train.bin":
        print(f"Train Tokens : {token_lengths}")
    else:
        print(f"Test Tokens : {token_lengths}")
    total_length = token_lengths.sum()

    # Prepare a memory-mapped file
    mmap = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_length,))

    idx = 0
    for _, text in tqdm(df['text'].iteritems(), total=df.shape[0]):
        tokens = encoder.encode_ordinary(text)
        mmap[idx: idx + len(tokens)] = tokens
        idx += len(tokens)

    mmap.flush()


# Tokenize and save the training and validation splits
tokenize_and_save(train_df, 'train.bin')
tokenize_and_save(val_df, 'val.bin')
