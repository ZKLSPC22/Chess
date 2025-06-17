import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_config, safe_merge, override_config
from env import ChessEnv


config = get_config('common')


# Technically, PPO is a offline and offpolicy training paradigm, however for importance sampling to work, we have to use data from a recent version of the agent, and intertwine training and data collection.
class PpoTrain:
    def __init__(self, agent, transition_dataset):
        class_name = self.__class__.__name__.lower()
        self.config = override_config(config, get_config(class_name))
        self.agent = agent
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
    def __init__(self, new_agent, old_agent):
        class_name = self.__class__.__name__.lower()
        self.config = override_config(config, get_config(class_name))
        self.new_agent = new_agent
        self.old_agent = old_agent
        self.data_set_format = ('state', 'action', 'old_log_prob', 'return', 'advantage')
        self.data_set = []
    
    def collect(self):
        new_agent_color = random.randint(0, 1)
        
        while len(self.data_set) < self.config['count']:
            episodes = []
            state = ChessEnv.initial_state()
            terminated = False
            
            while not terminated:
                agent = self.new_agent if state.color == new_agent_color else self.old_agent
                legal_actions = ChessEnv.get_legal_actions(state)
                if not legal_actions:
                    raise ValueError("Checkmate or stalemate undetected by the environment")
                
                with torch.no_grad():
                    policy_logits = agent(state.unsqueeze(0))[0].squeeze(0)
                    mask = torch.zeros_like(policy_logits, dtype=torch.bool)
                    mask[legal_actions] = True
                    masked_logits = policy_logits.masked_fill(~mask, float('-inf'))
                    probs = torch.softmax(masked_logits, dim=0)
                    action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log(probs[action])
                
                next_state, reward, terminated, truncated, info = ChessEnv.step(state, action)
                
                episodes.append((state, action, log_prob, reward))
                state = next_state
            
            # Calculate returns for each state in the episode
            returns = []
            R = 0.0  # Initialize return
            # Calculate returns backwards from end of episode
            for _, _, _, reward in reversed(episodes):
                R = reward + self.config['gamma'] * R  # Discounted return
                returns.insert(0, R) # Insert at the beginning of the list
            
            # Calculate advantages and store final transitions
            for (state, action, old_log_prob, _, value), ret in zip(episodes, returns):
                adv = ret - value.item()  # Advantage = return - value
                self.data_set.append((state, action, old_log_prob, ret, adv))
                
        return self.data_set


class PolicyValueTrain:
    def __init__(self, agent, dataset):
        class_name = self.__class__.__name__.lower()
        self.config = override_config(config, get_config(class_name))
        self.agent = agent
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.agent_output = ('policy_logits', 'value')
        self.data_set_format = ('state', 'policy_target', 'value_target')
        self.restrictions = set('offline', 'offpolicy', 'nn')

    def train(self):
        for epoch in range(self.config['epochs']):
            for states, pi_targets, z_targets in self.loader:
                policy_logits, values = self.agent(states)  # (B, 4672), (B, 1)

                policy_loss = F.cross_entropy(policy_logits, pi_targets.argmax(dim=1))  # pi_targets: (B, 4672)
                value_loss = F.mse_loss(values.squeeze(), z_targets)  # z_targets: (B,)

                l2_penalty = sum((p**2).sum() for p in self.agent.parameters())
                loss = policy_loss + self.config['value_coef'] * value_loss + self.config['weight_decay'] * l2_penalty

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
