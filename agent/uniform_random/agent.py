import random
import env


class UniformRandom:
    def __init__(self):
        self.config = {}

    def select_action(self, state):
        legal_actions = env.get_legal_actions(state)
        action = random.choice(legal_actions)
        return action
    