import random
import torch
import torch.nn.functional as F
import logging
from env import *
from torch.utils.data import DataLoader
from planning.mcts import MCTS
import torch.optim as optim


training_logger = logging.getLogger('training')


# Technically, PPO is a offline and offpolicy training paradigm, however for importance sampling to work, we have to use data from a recent version of the agent, and intertwine training and data collection.
class PpoTrain:
    def __init__(self, agent, transition_dataset, train_config, optimizer):
        self.config = train_config
        self.agent = agent
        self.optimizer = optimizer
        self.transition_dataset = transition_dataset
        self.loader = DataLoader(transition_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.agent_output = ('policy_logits', 'value')
        self.data_set_format = ('state', 'action', 'old_log_prob', 'return', 'advantage')
        self.restrictions = {'offline', 'offpolicy', 'nn'}

    def train(self):
        for epoch in range(self.config['epochs']):
            for states, actions, old_log_probs, returns, advantages in self.loader:
                policy_logits, values = self.agent.model(states)
            # Apply softmax to policy logits to get probabilities
            dist = torch.distributions.Categorical(logits=policy_logits)
            # Recalculate log probabilities to normalize the policy
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.config['clip_eps'], 1 + self.config['clip_eps']) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.agent


class PpoDataCollection:
    def __init__(self, new_agent, old_agent, data_collection_config):
        self.config = data_collection_config
        self.new_agent = new_agent
        self.old_agent = old_agent
        self.data_set_format = ('state', 'action', 'old_log_prob', 'return', 'advantage')
        self.data_set = []
    
    def collect(self):
        training_logger.info(f"Starting PPO data collection. Target: {self.config.get('count', 'N/A')} transitions.")
        new_agent_color = random.randint(0, 1)
        chess_env = ChessEnv()
        last_reported = 0
        game_counter = 0
        
        while len(self.data_set) < self.config['count']:
            game_counter += 1
            if hasattr(self.new_agent, 'mcts') and self.new_agent.mcts is not None:
                self.new_agent.mcts.reset()
            if hasattr(self.old_agent, 'mcts') and self.old_agent.mcts is not None:
                self.old_agent.mcts.reset()

            episodes = []
            state = chess_env.initial_state()
            terminated = False
            game_move_count = 0
            training_logger.info(f"Starting self-play game #{game_counter}. Current transitions: {len(self.data_set)}.")
            
            while not terminated:
                state_color = state[16, 0, 0].item()
                agent = self.new_agent if state_color == new_agent_color else self.old_agent
                agent_type = 'new_agent' if state_color == new_agent_color else 'old_agent'
                legal_actions = get_legal_actions(state)
                if not legal_actions:
                    raise ValueError("Checkmate or stalemate undetected by the environment")
                
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(agent.device)
                    policy_logits, value = agent.model(state_tensor)
                    policy_logits = policy_logits.squeeze(0)

                    mask = torch.zeros_like(policy_logits, dtype=torch.bool)
                    mask[legal_actions] = True
                    masked_logits = policy_logits.masked_fill(~mask, float('-inf'))
                    probs = torch.softmax(masked_logits, dim=0)
                    result = agent.select_action(state)
                    action = result if isinstance(result, int) else result[0]
                    log_prob = torch.log(probs[action])
                
                training_logger.debug(f"Game #{game_counter}, move {game_move_count+1}: {agent_type} selects action {action} from {len(legal_actions)} legal actions.")
                next_state, reward, terminated, truncated, info = chess_env.step(state, action)
                
                if hasattr(self.new_agent, 'mcts') and self.new_agent.mcts is not None:
                    self.new_agent.mcts.advance_tree(action)
                if hasattr(self.old_agent, 'mcts') and self.old_agent.mcts is not None:
                    self.old_agent.mcts.advance_tree(action)

                episodes.append((state, action, log_prob, reward, value))
                state = next_state
                game_move_count += 1
            
            training_logger.info(f"Self-play game #{game_counter} finished in {game_move_count} moves. Processing trajectory.")
            # Calculate returns for each state in the episode
            returns = []
            R = 0.0  # Initialize return
            # Calculate returns backwards from end of episode
            for _, _, _, reward, _ in reversed(episodes):
                R = reward + self.config['gamma'] * R  # Discounted return
                returns.insert(0, R) # Insert at the beginning of the list
            
            training_logger.debug(f"Processing {len(episodes)} transitions from game #{game_counter}.")
            # Calculate advantages and store final transitions
            for idx, ((state, action, old_log_prob, reward, value), ret) in enumerate(zip(episodes, returns)):
                adv = ret - value.item()  # Advantage = return - value
                self.data_set.append((state, action, old_log_prob, ret, adv))
                training_logger.debug(f"Game #{game_counter}, transition {idx+1}: action={action}, return={ret:.4f}, advantage={adv:.4f}.")
                # Periodic progress logging
                if len(self.data_set) - last_reported >= 100:
                    training_logger.info(f"Collected {len(self.data_set)} transitions so far...")
                    last_reported = len(self.data_set)
                
        training_logger.info(f"PPO data collection finished. Collected {len(self.data_set)} transitions.")
        return self.data_set


class ValuePolicyTrain:
    def __init__(self, agent, dataset, train_config, optimizer):
        self.config = train_config
        self.agent = agent
        self.optimizer = optimizer
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.agent_output = ('policy_logits', 'value')
        self.data_set_format = ('state', 'policy_target', 'value_target')
        self.restrictions = {'offline', 'offpolicy', 'nn'}

    def train(self):
        training_logger.info(f"Starting Policy/Value training for {self.config.get('epochs', 'N/A')} epochs.")
        for epoch in range(self.config['epochs']):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            for states, pi_targets, z_targets in self.loader:
                policy_logits, values = self.agent(states)  # (B, 4672), (B, 1)

                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = F.kl_div(log_probs, pi_targets, reduction='batchmean')
                value_loss = F.mse_loss(values.squeeze(), z_targets)  # z_targets: (B,)

                l2_penalty = sum((p**2).sum() for p in self.agent.parameters())
                loss = policy_loss + self.config['value_coef'] * value_loss + self.config['weight_decay'] * l2_penalty

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

            num_batches = len(self.loader)
            training_logger.debug(
                f"Epoch {epoch + 1}/{self.config['epochs']} | "
                f"Avg Policy Loss: {epoch_policy_loss / num_batches:.4f} | "
                f"Avg Value Loss: {epoch_value_loss / num_batches:.4f}"
            )
        training_logger.info("Policy/Value training finished.")


class ValuePolicyDataCollection:
    def __init__(self, agent, opponent, data_collection_config):
        self.config = data_collection_config
        self.agent = agent
        self.opponent = opponent
        self.data_set_format = ('state', 'policy_target', 'value_target')
        self.data_set = []
    
    def collect(self):
        training_logger.info(f"Starting Policy/Value data collection. Target: {self.config.get('count', 'N/A')} transitions.")
        chess_env = ChessEnv()
        last_reported = 0
        game_counter = 0
        
        while len(self.data_set) < self.config['count']:
            if hasattr(self.agent, 'mcts') and self.agent.mcts is not None:
                self.agent.mcts.reset()
            if hasattr(self.opponent, 'mcts') and self.opponent.mcts is not None:
                self.opponent.mcts.reset()

            episode_history = []
            state = chess_env.initial_state()
            terminated = False
            game_move_count = 0
            game_counter += 1
            # Randomly assign colors
            agent_color_is_white = random.choice([True, False])
            training_logger.info(f"Starting Policy/Value self-play game #{game_counter}. Current transitions: {len(self.data_set)}.")

            while not terminated:
                current_player_is_white = state[16, 0, 0].item() == 1
                
                if current_player_is_white == agent_color_is_white:
                    # Agent's turn: Use MCTS to get action and policy target
                    result = self.agent.select_action(state)
                    action = result if isinstance(result, int) else result[0]
                    episode_history.append({'state': state.clone(), 'policy': action, 'player_is_white': current_player_is_white})
                    agent_type = 'agent'
                else:
                    # Opponent's turn
                    result = self.opponent.select_action(state)
                    action = result if isinstance(result, int) else result[0]
                    agent_type = 'opponent'

                training_logger.debug(f"Game #{game_counter}, move {game_move_count+1}: {agent_type} selects action {action}.")
                state, _, terminated, _, _ = chess_env.step(state, action)
                
                if hasattr(self.agent, 'mcts') and self.agent.mcts is not None:
                    self.agent.mcts.advance_tree(action)
                if hasattr(self.opponent, 'mcts') and self.opponent.mcts is not None:
                    self.opponent.mcts.advance_tree(action)

                game_move_count += 1
            
            training_logger.info(f"Policy/Value self-play game #{game_counter} finished in {game_move_count} moves. Processing trajectory.")
            # Determine game outcome and assign value targets
            value_target = chess_env._get_reward(chess_env.board).item()

            training_logger.debug(f"Processing {len(episode_history)} transitions from game #{game_counter}.")
            # Store transitions with the correct value perspective
            for idx, data_point in enumerate(episode_history):
                s = data_point['state']
                pi = data_point['policy']
                player_is_white = data_point['player_is_white']
                
                # Value is from the perspective of the current player for that state
                z = value_target if player_is_white else -value_target
                self.data_set.append((s, pi, torch.tensor(z, dtype=torch.float32)))
                training_logger.debug(f"Game #{game_counter}, transition {idx+1}: action={pi}, value_target={z}.")
                
                # Periodic progress logging
                if len(self.data_set) - last_reported >= 100:
                    training_logger.info(f"Collected {len(self.data_set)} transitions so far...")
                    last_reported = len(self.data_set)

        training_logger.info(f"Policy/Value data collection finished. Collected {len(self.data_set)} transitions.")
        return self.data_set
