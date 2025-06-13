import os
import importlib
import re
import agent.mcts_ppo_pvl_resnet.agent as mcts_ppo_pvl_resnet
from agent import *


class TrainingParadigms:
    def __init__(self, instance):
        pass

# Agent names MUST be in snake_case matching the folder name
def create_agent_instance(agent_name):
    # module object
    agent_module = importlib.import_module(f'agent.{agent_name}.agent')
    
    agent_key = _normalize_name(agent_name)  # e.g., 'mctsppopvlresnet'
    agent_class_name = None

    # objects names as strings
    for name in dir(agent_module):
        # turn string into python object
        obj = getattr(agent_module, name)
        # check if obj is a class
        if isinstance(obj, type) and _normalize_name(name) == agent_key:
            agent_class_name = name
            break

    if agent_class_name is None:
        raise ValueError(f"Couldn't find a matching agent class in agent.{agent_name}.agent")

    agent_dir = os.path.join('agent', agent_name)
    # List all entries in agent_dir, and keep only those whose names fully match the pattern 'instance' followed by one or more digits
    instance_dirs = [
        d  # d is the name of a file or directory in agent_dir
        for d in os.listdir(agent_dir)  # iterate over all entries in agent_dir
        if re.fullmatch(r'instance\d+', d)  # include only if the name matches 'instance' + digits
    ]
    instance_nums = [int(d.replace('instance', '')) for d in instance_dirs]
    next_instance_num = max(instance_nums, default=-1) + 1
    instance_name = f'instance{next_instance_num}'
    instance_path = os.path.join(agent_dir, instance_name)
    os.makedirs(instance_path, exist_ok=True)
    
    AgentClass = getattr(agent_module, agent_class_name)
    agent = AgentClass()
    return agent, instance_path

def _normalize_name(name: str) -> str:
    """Convert snake_case or PascalCase to lowercase without separators."""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()
