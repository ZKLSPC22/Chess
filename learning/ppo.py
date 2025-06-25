import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from utils.misc import get_activation

# ========================================================================================
# == Special Notice: Integrated Model and Training Module                             ==
# ========================================================================================
# This file serves as an integrated module for the PPO (Proximal Policy Optimization)
# algorithm. It contains both the neural network architecture (`ResNet`) and the
# specific training logic (`PpoTrain`) that is tightly coupled with it.
#
# **Reasoning**: The PPO training process is highly specific. It depends on the
# exact outputs of the model (policy logits and value), requires a unique data
# format (including old log probabilities and advantages), and uses a custom
# clipped surrogate objective function.
#
# By keeping the model and its trainer together, we ensure that this complex,
# co-dependent logic is self-contained and easier to manage. This is a deliberate
# design choice for this specific algorithm and does not apply to more generic
# training methods, which should remain separate.
# ========================================================================================

training_logger = logging.getLogger('training')


class PpoTrain:
    def __init__(self, agent, transition_dataset, train_config, optimizer):
        self.agent = agent
        self.config = train_config
        self.clip_eps = self.config['clip_eps']
        self.device = agent.device
        self.optimizer = optimizer
        self.loader = DataLoader(transition_dataset, batch_size=self.config['batch_size'], shuffle=True)
        training_logger.info(f"PPO trainer initialized.")

    def train(self):
        self.agent.model.train()
        training_logger.info(f"Starting PPO training for {self.config['epochs']} epochs.")
        for epoch in range(self.config['epochs']):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            for states, actions, old_log_probs, returns, advantages in self.loader:
                # Move data to the same device as the model
                states = states.to(self.device)
                actions = actions.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)

                policy_logits, values = self.agent.model(states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

                value_loss = F.mse_loss(values.squeeze(), returns)
                entropy = dist.entropy().mean()

                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy

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
        training_logger.info("PPO training finished.")
        return self.agent


class ResBlock(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x.clone()
        x = self.bn1(self.conv1(x))
        x = self.activation(x)
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.activation(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(2)
        self.activation = activation
        self.fc = nn.Linear(128, 4672)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = activation
    
    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class ResNet(nn.Module):
    def __init__(self, model_config, in_channels=17):
        super(ResNet, self).__init__()
        
        # Unpack model configuration
        channels = model_config['channels']
        num_res_blocks = model_config['num_res_blocks']
        activation_name = model_config['activation']
        activation = get_activation(activation_name)

        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(channels, activation) for _ in range(num_res_blocks)])
        self.policy_head = PolicyHead(channels, activation)
        self.value_head = ValueHead(channels, activation)
        logging.debug("ResNet model initialized.")
        
    def forward(self, x):
        x = self.conv(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
