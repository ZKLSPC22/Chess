import os
import re
import importlib
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_activation(activation_name):
    """Returns an activation function class from its name."""
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
    }
    activation_cls = activations.get(activation_name.lower())
    if not activation_cls:
        raise ValueError(f"Unknown activation function: {activation_name}")
    logger.debug(f"Retrieved activation function: {activation_name}")
    return activation_cls()

def snake_to_pascal(name: str) -> str:
    """Converts snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

def normalize_name(name: str) -> str:
    """Converts any case format to lowercase without special chars."""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def create_agent_class_object(agent_name, agent_config):
    """Dynamically creates an agent class instance from its module."""
    logger.debug(f"Attempting to create class object for agent: {agent_name}")
    try:
        agent_module = importlib.import_module(f'agent.{agent_name}.agent')
        pascal_name = snake_to_pascal(agent_name)
        if hasattr(agent_module, pascal_name):
            AgentClass = getattr(agent_module, pascal_name)
            logger.debug(f"Found and loaded agent class '{pascal_name}' from module.")
            return AgentClass(agent_config)
        
        # Fallback for backward compatibility
        logger.warning(f"Could not find class '{pascal_name}', falling back to normalization.")
        agent_key = normalize_name(agent_name)
        for name in dir(agent_module):
            if isinstance(getattr(agent_module, name), type) and normalize_name(name) == agent_key:
                AgentClass = getattr(agent_module, name)
                logger.debug(f"Found and loaded agent class '{name}' via fallback.")
                return AgentClass(agent_config)
                
        available_classes = [name for name in dir(agent_module) if isinstance(getattr(agent_module, name), type)]
        logger.error(f"Could not find a matching agent class for '{agent_name}'. Available: {available_classes}")
        raise ValueError(
            f"Couldn't find a matching agent class in agent.{agent_name}.agent.\nAvailable classes: {available_classes}"
        )
    except ImportError:
        logger.critical(f"Failed to import agent module: agent.{agent_name}.agent", exc_info=True)
        raise

def generate_next_instance_dir(agent_name):
    """Generates the next available instance directory path."""
    agent_dir = os.path.join('agent', agent_name)
    logger.debug(f"Generating next instance directory in {agent_dir}")
    if not os.path.exists(agent_dir):
        logger.error(f"Agent directory not found: {agent_dir}")
        raise FileNotFoundError(f"Cannot create instance - {agent_dir} does not exist.")
        
    instance_dirs = [
        d for d in os.listdir(agent_dir)
        if os.path.isdir(os.path.join(agent_dir, d)) and re.fullmatch(r'instance\d+', d)
    ]
    instance_nums = [int(d.replace('instance', '')) for d in instance_dirs]
    next_instance_num = max(instance_nums, default=-1) + 1
    next_dir = os.path.join(agent_dir, f'instance{next_instance_num}')
    logger.debug(f"Next instance directory will be: {next_dir}")
    return next_dir
