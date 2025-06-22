import random
import torch
import torch.nn.functional as F
import logging
from env import ChessEnv
from torch.utils.data import DataLoader


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
        self.restrictions = set('offline', 'offpolicy', 'nn')

    def train(self):
        for epoch in range(self.config['epochs']):
            for states, actions, old_log_probs, returns, advantages in self.loader:
                policy_logits, values = self.agent(states)
            # Apply softmax to policy logits to get probabilities
            dist = torch.distributions.Categorical(logits=policy_logits)
            # Recalculate log probabilities to normalize the policy
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
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
        
        while len(self.data_set) < self.config['count']:
            episodes = []
            state = chess_env.initial_state()
            terminated = False
            game_move_count = 0
            
            while not terminated:
                state_color = state[16, 0, 0].item()
                agent = self.new_agent if state_color == new_agent_color else self.old_agent
                legal_actions = chess_env.get_legal_actions(state)
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
                    action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log(probs[action])
                
                next_state, reward, terminated, truncated, info = chess_env.step(state, action)
                
                episodes.append((state, action, log_prob, reward, value))
                state = next_state
                game_move_count += 1
            
            training_logger.debug(f"Self-play game finished in {game_move_count} moves. Processing trajectory.")
            # Calculate returns for each state in the episode
            returns = []
            R = 0.0  # Initialize return
            # Calculate returns backwards from end of episode
            for _, _, _, reward, _ in reversed(episodes):
                R = reward + self.config['gamma'] * R  # Discounted return
                returns.insert(0, R) # Insert at the beginning of the list
            
            # Calculate advantages and store final transitions
            for (state, action, old_log_prob, reward, value), ret in zip(episodes, returns):
                adv = ret - value.item()  # Advantage = return - value
                self.data_set.append((state, action, old_log_prob, ret, adv))
                
        training_logger.info(f"PPO data collection finished. Collected {len(self.data_set)} transitions.")
        return self.data_set


class PolicyValueTrain:
    def __init__(self, agent, dataset, train_config, optimizer):
        self.config = train_config
        self.agent = agent
        self.optimizer = optimizer
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.agent_output = ('policy_logits', 'value')
        self.data_set_format = ('state', 'policy_target', 'value_target')
        self.restrictions = set('offline', 'offpolicy', 'nn')

    def train(self):
        training_logger.info(f"Starting Policy/Value training for {self.config.get('epochs', 'N/A')} epochs.")
        for epoch in range(self.config['epochs']):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            for states, pi_targets, z_targets in self.loader:
                policy_logits, values = self.agent(states)  # (B, 4672), (B, 1)

                policy_loss = F.cross_entropy(policy_logits, pi_targets.argmax(dim=1))  # pi_targets: (B, 4672)
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
