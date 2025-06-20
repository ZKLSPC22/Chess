import torch
import torch.optim as optim
import learning.ppo as ppo
import planning.mcts as mcts
import os
from utils import safe_merge, override_config
import env


# New configurations are of higher priority
agent_config = {}
# Merge all inherited configs
config = safe_merge(ppo.config, mcts.config)
# Override configs
config = override_config(config, agent_config)

# Class name MUST match the agent name for importlib to work
class mcts_ppo_pvl_resnet:
    def __init__(self):
        self.model = ppo.ResNet(
            in_channels=17,
            channels=ppo.channels,
            num_res_blocks=ppo.num_res_blocks,
            activation=ppo.activation
            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=ppo.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.training = set('offline', 'offpolicy', 'nn')
        self.config = config

    def select_action(self, state):
        self.model.eval()
        with torch.no_grad():
            policy_logits, _ = self.model(state)
            legal_actions = env.get_legal_actions(state)
            for i, logit in enumerate(policy_logits):
                if i not in legal_actions:
                    policy_logits[i] = -1e10
            action_idx = torch.argmax(policy_logits, dim=1).item()
            return action_idx
