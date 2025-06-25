import torch
import learning.ppo as ppo
from planning.mcts import MCTS
import logging


logger = logging.getLogger(__name__)


# Class name MUST match the agent name for importlib to work
class MctsPpoPvlResnet:
    def __init__(self, instance_config):
        logger.info("Initializing MctsPpoPvlResnet agent.")
        self.config = instance_config
        # Model config
        model_config = self.config['ppo']['model']
        self.model = ppo.ResNet(model_config=model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"MctsPpoPvlResnet agent model moved to device: {self.device}")
        
        # MCTS planner
        self.mcts = MCTS(self, self.config['mcts'])

    def select_action(self, state):
        logger.debug("MctsPpoPvlResnet selecting action for given state tensor.")
        self.model.eval()
        action = self.mcts.select_action(state)
        logger.debug(f"Agent selected action index: {action}")
        return action
