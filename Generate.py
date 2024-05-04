from config import get_config, get_weights_file_path
from TransformerModel import build_transformer
import torch
import tiktoken


def generate_song(seed_text: str, num_samples, max_length: int = 1500):
    # Define the device and load configurations
    vocab_src_len = 50304
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    config = get_config()

    # Initialize the tokenizer and model
    tokenizer = tiktoken.get_encoding("gpt2")
    model = build_transformer(vocab_src_len, config['seq_len'], config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = get_weights_file_path(config)  # Update function name as needed
    state = torch.load(model_filename)
    state_dic = state['model_state_dict']
    model.load_state_dict(state_dic)

    model.eval()

    # Encode the starting string to token IDss
    print(f"Input: {seed_text}")
    input = tokenizer.encode_ordinary(seed_text)
    out_id = tokenizer.encode_ordinary("[EOS]")
    input = (torch.tensor(input, dtype=torch.long, device=device)[None, ...])

    generated_outputs = ""

    # Run generation
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(config, input, device, max_length, tokenizer)
            # Decoding and storing the output in the dictionary
            generated_outputs += f'Sample {k + 1}:\n' + tokenizer.decode(y[0].tolist()) + '\n\n'

    print(f"\n\nGenerated Output:\n{generated_outputs}")
    # y = model.generate(config, input, max_length)
    # print(f"\n\nGenerated Output:\n{tokenizer.decode(y[0].tolist())}")
    return generated_outputs


# Example use
# generate_song(get_config()['generate_input'], 10, 1000)

