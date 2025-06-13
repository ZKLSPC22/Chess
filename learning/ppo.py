import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config_loader import get_config
from utils import *
from torch.utils.data import DataLoader


# Load corresponding config dictionary
config = get_config(__file__)
'''
This is doing:
for key, value in config.items():
    globals()[key] = value
'''
globals().update(config)


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
        self.conv = nn.Conv2d(channels, 2)
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
    def __init__(self, in_channels=17, channels=channels, num_res_blocks=num_res_blocks, activation=activation):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(channels, activation) for _ in range(num_res_blocks)])
        self.policy_head = PolicyHead(channels, activation)
        self.value_head = ValueHead(channels, activation)
        
    def forward(self, x):
        x = self.conv(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


def train(agent, optimizer, transition_dataset, clip_eps=clip_eps, entropy_coef=entropy_coef, value_coef=value_coef, epochs=epochs, batch_size=batch_size):
    loader = DataLoader(transition_dataset, batch_size=batch_size, shuffle=True)
    agent.train()

    for epoch in range(epochs):
        for states, actions, old_log_probs, returns, advantages in loader:
            policy_logits, values = agent(states)
            # Apply softmax to policy logits to get probabilities
            dist = torch.distributions.Categorical(logits=policy_logits)
            # Recalculate log probabilities to normalize the policy
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
