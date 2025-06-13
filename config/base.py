from config_loader import get_config

class Configurable:
    def __init__(self, config_key=None):
        self.config_key = config_key or self.__class__.__name__.lower()
        self.config = get_config(self.config_key)