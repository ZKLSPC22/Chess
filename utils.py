import torch
import torch.nn as nn
import yaml
import chess
import os


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.01)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softmax':
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

_config = None

def _load_config():
    global _config
    if _config is None:
        with open('config.yaml', 'r') as f:
            # yaml.safe_load() maps yaml object to python object, in our case, a mapping is mapped to a dict
            _config = yaml.safe_load(f)
    return _config

# This method works for both file paths and class names
def get_config(file_path):
    _config = _load_config()
    # Example: "RL/leaning/ppo.py" -> "ppo.py"
    filename = os.path.basename(file_path)
    # Example: "ppo.py" -> "ppo"
    # splitext("a.b.c") -> ("a.b", ".c")
    config_key = os.path.splitext(filename)[0].lower()
    # returns value of config_key in config (a dict), if not found, returns {}
    return _config.get(config_key, {})

def safe_merge(*configs):
    merged = {}
    for config in configs:
        for k, v in config.items():
            if k in merged:
                raise ValueError(f"Duplicate key {k} in configs")
            merged[k] = v
    return merged

def override_config(config, override_config):
    for k, v in override_config.items():
        config[k] = v
    return config

def action_index_to_uci(action_index, chess_env):
    from_square = action_index // 73
    to_promo = action_index % 73
    if to_promo < 56:
        to_square = to_promo
        promotion = None
    else:
        to_square = (to_promo - 56) // 4 + (from_square // 8 == 6) * 8
        promo_type = (to_promo - 56) % 4
        promotion = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promo_type]
    return chess.Move(from_square, to_square, promotion=promotion).uci()
