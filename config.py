from pathlib import Path
def get_config():
    return {
        "batch_size": 12,
        "num_epochs": 5,
        "lr": 1e-3, # 10**-4,  1e-3   3e-4
        "seq_len": 64,
        "segment_len": 65,
        "max_seq_len": 1204,
        "d_model": 384,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": 'latest',
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "path": 'data/spotify_millsongdata.csv',
        "generate_input": f"Title: Watching Over Me; Tag: pop; Artist: Canadian Tenors;\n\n" + "Lyrics: \n"
    }

def get_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

