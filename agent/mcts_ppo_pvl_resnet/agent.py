import torch
import torch.optim as optim
import learning.ppo as ppo
from planning.mcts import MCTS
import logging
from utils.config import override_config

logger = logging.getLogger(__name__)

# Class name MUST match the agent name for importlib to work
class MctsPpoPvlResnet:
    def __init__(self, agent_config):
        logger.info("Initializing MctsPpoPvlResnet agent.")
        self.config = agent_config
        # Model config
        model_config = self.config['model']
        self.model = ppo.ResNet(model_config=model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"MctsPpoPvlResnet agent model moved to device: {self.device}")
        
        # MCTS planner
        self.mcts = MCTS(self, self.config['mcts'])

    def select_action(self, state, chess_env, return_policy=False):
        logger.debug(f"MctsPpoPvlResnet selecting action for board FEN: {chess_env.board.fen()}")
        self.model.eval()
        action = self.mcts.select_action(state, chess_env, return_policy)
        logger.debug(f"Agent selected action index: {action}")
        return action
