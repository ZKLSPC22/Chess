import random
import torch


class UniformRandom:
    def __init__(self):
        self.config = {}

    def select_action(self, state, chess_env, return_policy=False):
        legal_actions = chess_env.get_legal_actions(state)
        action = random.choice(legal_actions)

        if return_policy:
            policy = torch.zeros(4672, dtype=torch.float32)
            if legal_actions:
                uniform_prob = 1.0 / len(legal_actions)
                policy[legal_actions] = uniform_prob
            return action, policy
        
        return action
    