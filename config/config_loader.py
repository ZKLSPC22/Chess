import yaml
import os


_config = None

def _load_config():
    global _config
    if _config is None:
        with open('config.yaml', 'r') as f:
            # yaml.safe_load() maps yaml object to python object, in our case, a mapping is mapped to a dict
            _config = yaml.safe_load(f)
    return _config

def get_config(file_path):
    config = _load_config()
    # Example: "RL/leaning/ppo.py" -> "ppo.py"
    filename = os.path.basename(file_path)
    # Example: "ppo.py" -> "ppo"
    # splitext("a.b.c") -> ("a.b", ".c")
    config_key = os.path.splitext(filename)[0]
    # returns value of config_key in config (a dict), if not found, returns {}
    return config.get(config_key, {})
