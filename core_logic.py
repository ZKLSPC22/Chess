import pickle  # For serializing agent instances
import os      # File/directory operations
import logging
import env
import random
import chess
import time
import copy
import agent.mcts_ppo_pvl_resnet.agent as mcts_ppo_pvl_resnet  # Specific agent implementation
from agent import *
from training_data_collection import PpoDataCollection
from learning.ppo import PpoTrain
from utils.config import override_config, dump_instance_config, retrieve_instance_config
from utils.misc import create_agent_class_object, generate_next_instance_dir

logger = logging.getLogger(__name__)
eval_logger = logging.getLogger('evaluation')
training_logger = logging.getLogger('training')

class TrainingParadigms:
    """Manages different training approaches"""
    def __init__(self, agent_instance, instance_dir):
        self.agent_instance = agent_instance
        self.instance_dir = instance_dir

    def list_paradigms(self):
        """Returns a list of all available training paradigms"""
        paradigms = []
        for method_name in dir(self):
            if not method_name.startswith('_') and callable(getattr(self, method_name)) and method_name != 'list_paradigms':
                paradigms.append(method_name)
        return paradigms

    def ppo_self_play(self):
        # Creates two agent instances for self-play training
        training_logger.info(f"Starting PPO self-play round for instance: {self.instance_dir}")
        old_agent_instance = copy.deepcopy(self.agent_instance)
        
        # Collect training data through self-play
        training_logger.info("Collecting training data...")
        data_collector = PpoDataCollection(
            self.agent_instance,
            old_agent_instance,
            self.agent_instance.config['ppo_data_collection']
        )
        collected_data = data_collector.collect()
        training_logger.info(f"Data collection complete. Collected {len(collected_data)} transitions.")
        
        # Train and save new agent
        training_logger.info("Starting training...")
        trainer = PpoTrain(self.agent_instance, collected_data)
        new_agent_instance = trainer.train()
        training_logger.info("Training complete.")

        override_agent_instance(new_agent_instance, self.instance_dir)
        self.agent_instance = new_agent_instance # Update in-memory agent

    def multi_round_ppo_self_play(self, rounds=5):
        training_logger.info(f"Starting multi-round PPO self-play for {rounds} rounds.")
        for i in range(rounds):
            training_logger.info(f"--- Round {i+1}/{rounds} ---")
            self.ppo_self_play()
        training_logger.info("Multi-round PPO self-play finished.")


# Play a game between a human and an agent
def vs_human(agent_instance, human_color=None):
    chess_env = env.ChessEnv()
    if human_color is None:
        human_color = random.choice([chess.WHITE, chess.BLACK])
    elif human_color not in [chess.WHITE, chess.BLACK]:
        raise ValueError("Invalid human color")
    
    state = chess_env.initial_state()
    eval_logger.info(f"Starting new game: Human vs. Agent ({agent_instance.__class__.__name__})")
    eval_logger.info(f"Human is playing as {'WHITE' if human_color == chess.WHITE else 'BLACK'}")
    
    # Color codes for text
    BLUE_COLOR = '\033[1;34m'  # Bright blue
    RED_COLOR = '\033[1;31m'   # Bright red
    RESET_COLOR = '\033[0m'    # Reset color
    
    # Welcome message
    print("\n" + "="*50)
    print("CHESS GAME - HUMAN vs AGENT")
    print("="*50)
    print(f"You are playing as: {BLUE_COLOR}BLUE{RESET_COLOR}" if human_color else f"You are playing as: {RED_COLOR}RED{RESET_COLOR}")
    print(f"Agent is playing as: {RED_COLOR}RED{RESET_COLOR}" if human_color else f"Agent is playing as: {BLUE_COLOR}BLUE{RESET_COLOR}")
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
            eval_logger.info(f"Human played move: {action}")
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
            eval_logger.info(f"Agent played move: {ai_move}")
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
        eval_logger.info(f"Game over: Checkmate. Winner: {'Human' if chess_env.board.turn != human_color else 'Agent'}")
    elif chess_env.board.is_stalemate():
        print(f"\nSTALEMATE! It's a draw!")
        eval_logger.info("Game over: Stalemate.")
    else:
        print(f"\nDRAW! Game ended in a draw!")
        eval_logger.info("Game over: Draw by other condition.")
    
    print("Thanks for playing!")

# Play a game between two agents with rendering, the first agent is assumed to be white
def vs_agent_with_render(agent_instance1, agent_instance2):
    chess_env = env.ChessEnv()
    state = chess_env.initial_state()
    eval_logger.info(f"Starting new game: {agent_instance1.__class__.__name__} (W) vs. {agent_instance2.__class__.__name__} (B)")

    # Color codes for text
    BLUE_COLOR = '\033[1;34m'  # Bright blue
    RED_COLOR = '\033[1;31m'   # Bright red
    RESET_COLOR = '\033[0m'    # Reset color
    
    # Welcome message
    print("\n" + "="*50)
    print("CHESS GAME - AGENT vs AGENT")
    print("="*50)
    print(f"Agent 1 is playing as: {BLUE_COLOR}BLUE{RESET_COLOR}")
    print(f"Agent 2 is playing as: {RED_COLOR}RED{RESET_COLOR}")
    print("="*50)
    time.sleep(1.0)

    while not chess_env.board.is_game_over():
        # Show the board first
        chess_env.render(chess.WHITE)
        control = input("Press Enter to continue, or 'q' to quit")
        if control == 'q':
            break
        time.sleep(0.5)

        if chess_env.board.turn == chess.WHITE:
            action = agent_instance1.select_action(state, chess_env)
            action_uci = chess_env.action_index_to_uci(action)
            print(f"Agent 1 played: {action_uci}")
            eval_logger.debug(f"Agent 1 selected action: {action_uci}")
        else:
            action = agent_instance2.select_action(state, chess_env)
            action_uci = chess_env.action_index_to_uci(action)
            print(f"Agent 2 played: {action_uci}")
            eval_logger.debug(f"Agent 2 selected action: {action_uci}")

        state, _, _, _, _ = chess_env.step(state, action)
        print("-"*50)
    
    # Game over
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    time.sleep(0.5)

    # Final board state
    chess_env.render(chess.WHITE)
    time.sleep(0.5)
    
    # Determine winner
    if chess_env.board.is_checkmate():
        winner = "BLUE" if chess_env.board.turn == chess.BLACK else "RED"
        winner_color = BLUE_COLOR if chess_env.board.turn == chess.BLACK else RED_COLOR
        print(f"\nCHECKMATE! {winner_color}{winner}{RESET_COLOR} wins!")
        eval_logger.info(f"Game over: Checkmate. Winner: {'Agent 1 (W)' if chess_env.board.turn == chess.BLACK else 'Agent 2 (B)'}")
    elif chess_env.board.is_stalemate():
        print(f"\nSTALEMATE! It's a draw!")
        eval_logger.info("Game over: Stalemate.")
    else:
        print(f"\nDRAW! Game ended in a draw!")
        eval_logger.info("Game over: Draw by other condition.")

    print("Thanks for watching!")

# Agent names MUST be in snake_case matching the folder name
def create_agent_instance(agent_name, agent_config, config_dict=None):
    """
    Creates and saves a new agent instance. The passed `agent_config` is the
    final, resolved configuration for this new agent.
    
    Configuration Flow:
    1. This function receives `agent_config`, which has already been resolved in
       `main.py` by merging defaults with agent-specific settings from the main
       `config.yaml`.
    2. The agent object is created using this final configuration.
    3. The agent's final configuration is then saved to the new instance's
       local `config.yaml` to make it self-contained.
    
    Returns: (agent_instance, instance_dir)
    """
    logger.info(f"Request to create new agent instance for agent type: {agent_name}")
    if config_dict is None:
        config_dict = {}
        
    agent_class_object = create_agent_class_object(agent_name, agent_config)
    instance_dir = generate_next_instance_dir(agent_name)
    
    # Create directory and save agent instance
    os.makedirs(instance_dir, exist_ok=True)
    with open(os.path.join(instance_dir, 'instance.pkl'), 'wb') as f:
        pickle.dump(agent_class_object, f)
    
    # The config passed in is already the final, resolved config.
    # We dump this to the instance's config file.
    dump_instance_config(instance_dir, agent_class_object.config)
    logger.info(f"Agent '{agent_name}' instance created successfully at: {instance_dir}")
    
    return agent_class_object, instance_dir

def override_agent_instance(agent, instance_dir, config_dict=None):
    """Overwrites existing agent with new instance"""
    logger.info(f"Overwriting agent instance at: {instance_dir}")
    if config_dict is None:
        config_dict = {}

    if not os.path.exists(instance_dir):
        logger.error(f"Attempted to override non-existent instance directory: {instance_dir}")
        raise FileNotFoundError(f"{instance_dir} does not exist.")
    
    # Save new agent instance
    with open(os.path.join(instance_dir, 'instance.pkl'), 'wb') as f:
        pickle.dump(agent, f)

    # The agent's config is the source of truth, optionally overridden by a passed dict
    final_config = override_config(config_dict, agent.config)
    dump_instance_config(instance_dir, final_config)
    logger.info(f"Agent instance at {instance_dir} has been overridden.")

def load_instance(instance_dir, base_config):
    """
    Loads an agent instance and applies the latest configuration hierarchy.
    
    Configuration Flow:
    1. Receives `base_config`, which contains the latest defaults and agent-specific
       settings from the main `config.yaml`. This ensures that any updates to the
       main config are considered when an old instance is loaded.
    2. Loads the pickled agent object from the instance directory.
    3. Applies the `base_config` to the loaded agent.
    4. Loads the instance's own local `config.yaml`, which contains any
       fine-tuning overrides specific to this instance.
    5. Applies these local overrides on top of the `base_config`.
    
    This layered approach ensures that old instances are always up-to-date with
    the project's latest default settings while preserving their specific tuning.
    """
    logger.info(f"Loading agent instance from: {instance_dir}")
    instance_path = os.path.join(instance_dir, 'instance.pkl')
    if not os.path.exists(instance_path):
        logger.error(f"Instance file not found at: {instance_path}")
        raise FileNotFoundError(f"{instance_path} does not exist.")
    with open(instance_path, 'rb') as f:
        agent_instance = pickle.load(f)
    logger.debug(f"Successfully loaded instance object from {instance_path}")
    
    # 1. Start with the latest default configuration from the main config file.
    agent_instance.config = base_config
    
    # 2. Load the instance-specific config file.
    instance_specific_config = retrieve_instance_config(instance_dir)
    
    # 3. Apply the instance-specific tweaks on top of the latest defaults.
    agent_instance.config = override_config(instance_specific_config, agent_instance.config)
    
    logger.info(f"Agent instance from {instance_dir} loaded and configured successfully.")
    return agent_instance
