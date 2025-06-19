import torch
import torch.optim as optim
import learning.ppo as ppo 
import env
from utils import safe_merge, override_config


agent_config = {}
config = override_config(ppo.config, agent_config)

# Class name MUST match the agent name for importlib to work
class PpoResnet:
    def __init__(self):
        self.model = ppo.ResNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.config = config

    def select_action(self, state, chess_env):
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension and move to device
            state = state.unsqueeze(0).to(self.device)
            policy_logits, _ = self.model(state)
            # Remove batch dimension for the output
            policy_logits = policy_logits.squeeze(0)
            legal_actions = chess_env.get_legal_actions(state.squeeze(0))
            for i, logit in enumerate(policy_logits):
                if i not in legal_actions:
                    policy_logits[i] = -1e10
            action_idx = torch.argmax(policy_logits, dim=0).item()
            return action_idx
