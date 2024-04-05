from pathlib import Path
def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 1,
        "lr": 1e-3, #3e-4, # 10**-4,
        "seq_len": 300,
        "segment_len": 301,
        "max_seq_len": 1204,
        "d_model": 512,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "path": 'data/spotify_millsongdata.csv',
        "generate_input": "My heart is falling in love"
    }

def get_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

