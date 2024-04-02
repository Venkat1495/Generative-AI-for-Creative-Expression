from pathlib import Path
def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 1,
        "lr": 10**-4,
        "seq_len": 300,
        "d_model": 512,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "path": 'data/spotify_millsongdata.csv'
    }

def get_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

