import torch
import torch.nn as nn


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
    

def check_compatibility(agent, training, data):
    if training.policy.issubset(agent.training) and data.structure.issubset(agent.data):
        return True
    return False


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
            


