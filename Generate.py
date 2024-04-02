from pathlib import Path
from config import get_config, get_weights_file_path
from TransformerModel import build_transformer
from tokenizers import Tokenizer
import torch
from tqdm import tqdm


def generate_song(seed_text: str = "[SOS]", max_length: int = 300):
    # Define the device and load configurations
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    config = get_config()

    # Initialize the tokenizer and model
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    model = build_transformer(tokenizer.get_vocab_size(), config['seq_len'], config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = get_weights_file_path(config)  # Update function name as needed
    state = torch.load(model_filename)
    state_dic = state['model_state_dict']
    model.load_state_dict(state_dic)

    model.eval()
    with torch.no_grad():
        # Encode the seed text
        encoded_seed = tokenizer.encode(seed_text).ids
        input_ids = torch.tensor([tokenizer.token_to_id('[SOS]')] + encoded_seed, dtype=torch.int64).unsqueeze(0).to(device)

        # Generation loop
        for _ in range(max_length - len(encoded_seed)):

            # mask = (input_ids != tokenizer.token_to_id('[PAD]')).unsqueeze(0).int() & causal_mask(input_ids.size(1), device) # Update or create a mask if your model requires it
            mask = torch.tril(torch.ones((1, input_ids.size(1), input_ids.size(1))), diagonal=1).type(torch.int).to(device)

            output = model.decode(input_ids, mask, lambda msg: tqdm.write(msg))  # Adjust this call based on your model's method signature
            proj_output = model.project(output[:, -1, :], lambda msg: tqdm.write(msg))  # Adjust if necessary to match your model's output format
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


def causal_mask(size, device):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int).to(device)
    return mask == 0
# Example use
generate_song("", 50)

