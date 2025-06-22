import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_main_config(config_path='config.yaml'):
    """Loads the main configuration file."""
    logger.info(f"Loading main configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Recursive self reference for arbitrarily deep nested dicts
def override_config(new_config, base_config):
    """
    Recursively overrides base_config with values from new_config.
    This is useful for applying instance-specific configs on top of agent-defaults.
    """
    for key, value in new_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            base_config[key] = override_config(value, base_config[key])
        else:
            base_config[key] = value
    return base_config

def retrieve_instance_config(instance_dir):
    """Retrieves the configuration for a specific agent instance."""
    config_path = os.path.join(instance_dir, 'config.yaml')
    logger.debug(f"Retrieving instance config from {config_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Instance config file not found at {config_path}. Using agent's default config.")
        return {}

def dump_instance_config(instance_dir, config_dict):
    """Saves the configuration for a specific agent instance."""
    config_path = os.path.join(instance_dir, 'config.yaml')
    logger.debug(f"Dumping instance config to {config_path}")
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
