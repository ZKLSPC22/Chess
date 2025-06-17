import pickle  # For serializing agent instances
import os      # File/directory operations
import importlib  # Dynamic module importing
import re      # Regular expressions for name normalization
import yaml    # YAML config file handling
import env as env
import random
import chess
import time
import agent.mcts_ppo_pvl_resnet.agent as mcts_ppo_pvl_resnet  # Specific agent implementation
from agent import *
from training_data_collection import *


class TrainingParadigms:
    """Manages different training approaches"""
    def __init__(self, instance):
        pass

    def ppo_self_play(self, agent_instance):
        # Creates two agent instances for self-play training
        old_agent_instance, _ = create_agent_instance('ppo_resnet')
        # Collect training data through self-play
        PpoDataCollection(agent_instance, old_agent_instance)
        # Train and save new agent
        new_agent_instance = PpoTrain(agent_instance, old_agent_instance).train()
        override_agent_instance(new_agent_instance, agent_instance)


class DataCollectionParadigms:
    """Manages data collection strategies"""
    def __init__(self, instance):
        pass

    def ppo_data_collection(self):
        pass  # Placeholder for PPO data collection implementation


class MixedParadigm:
    """Combines multiple training approaches"""
    def __init__(self, instance):
        pass


def play_game(agent_instance, human_color=None):
    env = env.ChessEnv()
    if human_color is None:
        human_color = random.choice([chess.WHITE, chess.BLACK])
    elif human_color not in [chess.WHITE, chess.BLACK]:
        raise ValueError("Invalid human color")
    state = env.initial_state()
    while not env.board.is_game_over():
        if env.board.turn == human_color:
            action = input("Enter your move: ")
            action = env.encode_uci_to_action_index(action)
            state, _, _, _, _ = env.step(state, action)
        else:
            time.sleep(0.5)
            action = agent_instance.select_action(state)
            state, _, _, _, _ = env.step(state, action)
        env.render(human_color)
    if env.board.is_checkmate():
        print(f"Game over! {human_color} wins!")
    else:
        print("Game over! It's a draw!")


# Agent names MUST be in snake_case matching the folder name
def create_agent_instance(agent_name, config_dict=None):
    """
    Creates and saves a new agent instance
    Returns: (agent_instance, instance_dir)
    """
    agent_class_object = _create_agent_class_object(agent_name)
    instance_dir = _generate_next_instance_dir(agent_name)
    
    # Create directory and save agent instance
    os.makedirs(instance_dir, exist_ok=True)
    with open(os.path.join(instance_dir, 'instance.pkl'), 'wb') as f:
        pickle.dump(agent_class_object, f)
    
    # Save config if provided
    if config_dict:
        with open(os.path.join(instance_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

    return agent_class_object, instance_dir

def override_agent_instance(agent, instance_dir, config_dict=None):
    """Overwrites existing agent with new instance"""
    if not os.path.exists(instance_dir):
        raise FileNotFoundError(f"{instance_dir} does not exist.")
    
    # Save new agent instance
    with open(os.path.join(instance_dir, 'instance.pkl'), 'wb') as f:
        pickle.dump(agent, f)

    # Update config if provided
    if config_dict:
        with open(os.path.join(instance_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

def _create_agent_class_object(agent_name):
    """Dynamically creates agent instance from module"""
    # Import agent module
    agent_module = importlib.import_module(f'agent.{agent_name}.agent')
    agent_key = _normalize_name(agent_name)
    
    # Find matching class in module
    agent_class_name = None
    for name in dir(agent_module):
        if isinstance(getattr(agent_module, name), type) and _normalize_name(name) == agent_key:
            agent_class_name = name
            break
            
    # Error handling for missing class
    if agent_class_name is None:
        available_classes = [name for name in dir(agent_module) if isinstance(getattr(agent_module, name), type)]
        raise ValueError(
            f"Couldn't find a matching agent class in agent.{agent_name}.agent.\nAvailable classes: {available_classes}"
        )

    # Create and return instance
    AgentClass = getattr(agent_module, agent_class_name)
    return AgentClass()

def _generate_next_instance_dir(agent_name):
    """Generates next available instance directory path"""
    agent_dir = os.path.join('agent', agent_name)
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(f"Cannot create instance - {agent_dir} does not exist.")
        
    # Find all existing instance directories
    instance_dirs = [
        d for d in os.listdir(agent_dir)
        if os.path.isdir(os.path.join(agent_dir, d)) and re.fullmatch(r'instance\d+', d)
    ]
    # Calculate next instance number
    instance_nums = [int(d.replace('instance', '')) for d in instance_dirs]
    next_instance_num = max(instance_nums, default=-1) + 1
    return os.path.join(agent_dir, f'instance{next_instance_num}')

def _normalize_name(name: str) -> str:
    """Converts any case format to lowercase without special chars"""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()
