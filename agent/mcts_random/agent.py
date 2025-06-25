import random
import torch
import logging

logger = logging.getLogger(__name__)

class DummyModel:
    def __init__(self, device):
        self.device = device
    
    def __call__(self, state):
        # Return a uniform policy and a neutral value
        return torch.ones(4672) / 4672, 0.0

# Class name MUST match the agent name
class MctsRandom:
    def __init__(self, agent_config):
        self.config = agent_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # The MCTS planner expects the agent to have a 'model' attribute
        self.model = DummyModel(self.device)

    def select_action(self, state, chess_env, return_policy=False):
        logger.debug("MctsRandom selecting a random action.")
        legal_actions = chess_env.get_legal_actions(state)
        action = random.choice(legal_actions)
        if return_policy:
            # Return a uniform policy for consistency
            policy = torch.zeros(4672)
            policy[legal_actions] = 1.0 / len(legal_actions)
            return action, policy
        return action
    