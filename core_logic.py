import pickle  # For serializing agent instances
import os      # File/directory operations
import importlib  # Dynamic module importing
import re      # Regular expressions for name normalization
import yaml    # YAML config file handling
import env
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
    chess_env = env.ChessEnv()
    if human_color is None:
        human_color = random.choice([chess.WHITE, chess.BLACK])
    elif human_color not in [chess.WHITE, chess.BLACK]:
        raise ValueError("Invalid human color")
    
    state = chess_env.initial_state()
    
    # Color codes for text
    BLUE_COLOR = '\033[1;34m'  # Bright blue
    RED_COLOR = '\033[1;31m'   # Bright red
    RESET_COLOR = '\033[0m'    # Reset color
    
    # Welcome message
    print("\n" + "="*50)
    print("CHESS GAME - HUMAN vs AI")
    print("="*50)
    print(f"You are playing as: {BLUE_COLOR}BLUE{RESET_COLOR}" if human_color else f"You are playing as: {RED_COLOR}RED{RESET_COLOR}")
    print(f"AI is playing as: {RED_COLOR}RED{RESET_COLOR}" if human_color else f"AI is playing as: {BLUE_COLOR}BLUE{RESET_COLOR}")
    print("="*50)
    time.sleep(0.5)
    
    while not chess_env.board.is_game_over():
        # Show the board first
        chess_env.render(human_color)
        time.sleep(0.5)
        
        if chess_env.board.turn == human_color:
            # Human's turn
            current_player = "BLUE" if human_color else "RED"
            current_color = BLUE_COLOR if human_color else RED_COLOR
            print(f"\nYour turn ({current_color}{current_player}{RESET_COLOR})")
            time.sleep(0.5)
            print("Enter move (e.g., e2e4) or 'help' for examples:")
            time.sleep(0.3)
            
            while True:
                action = input("Move: ").strip().lower()
                
                if action == 'help':
                    print("\nMove examples:")
                    print("  e2e4    - Move pawn from e2 to e4")
                    print("  d7d5    - Move pawn from d7 to d5")
                    print("  g1f3    - Move knight from g1 to f3")
                    print("  e7e8q   - Pawn promotion to queen")
                    time.sleep(0.5)
                    continue
                elif not action:
                    print("Please enter a move!")
                    time.sleep(0.3)
                    continue
                
                try:
                    action_idx = chess_env.encode_uci_to_action_index(action)
                    legal_actions = chess_env.get_legal_actions(state)
                    if action_idx not in legal_actions:
                        print(f"Invalid move: {action}")
                        time.sleep(0.3)
                        continue
                    break
                except Exception as e:
                    print(f"Invalid format: {action}. Use format like 'e2e4'")
                    time.sleep(0.3)
                    continue
            
            print(f"You played: {action}")
            time.sleep(0.5)
            state, _, _, _, _ = chess_env.step(state, action_idx)
            
        else:
            # AI's turn
            current_player = "BLUE" if not human_color else "RED"
            current_color = BLUE_COLOR if not human_color else RED_COLOR
            print(f"\nAI turn ({current_color}{current_player}{RESET_COLOR})")
            time.sleep(0.5)
            print("AI is thinking...")
            time.sleep(0.5)
            
            action = agent_instance.select_action(state, chess_env)
            
            # Convert action back to UCI for display
            from_square, to_square, promotion = chess_env._decode_action_index(action)
            ai_move = chess.Move(from_square, to_square, promotion=promotion).uci()
            
            print(f"AI played: {ai_move}")
            time.sleep(0.5)
            state, _, _, _, _ = chess_env.step(state, action)
        
        # Turn separator
        print("-" * 50)
    
    # Game over
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    time.sleep(0.5)
    
    # Final board state
    chess_env.render(human_color)
    time.sleep(0.5)
    
    # Determine winner
    if chess_env.board.is_checkmate():
        winner = "BLUE" if chess_env.board.turn == chess.BLACK else "RED"
        winner_color = BLUE_COLOR if chess_env.board.turn == chess.BLACK else RED_COLOR
        print(f"\nCHECKMATE! {winner_color}{winner}{RESET_COLOR} wins!")
    elif chess_env.board.is_stalemate():
        print(f"\nSTALEMATE! It's a draw!")
    else:
        print(f"\nDRAW! Game ended in a draw!")
    
    print("Thanks for playing!")


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
