import random


class UniformRandom:
    def __init__(self):
        self.config = {}

    def select_action(self, state, chess_env):
        legal_actions = chess_env.get_legal_actions(state)
        return random.choice(legal_actions)
    