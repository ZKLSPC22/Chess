import torch
import logging
import learning.ppo as ppo 

logger = logging.getLogger(__name__)

# Class name MUST match the agent name for importlib to work
class PpoResnet:
    def __init__(self, agent_config):
        logger.info("Initializing PpoResnet agent.")
        self.config = agent_config
        self.model = ppo.ResNet(model_config=self.config['ppo']['model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"PpoResnet agent model moved to device: {self.device}")

    def select_action(self, state, chess_env):
        logger.debug(f"PpoResnet selecting action for board FEN: {chess_env.board.fen()}")
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension and move to device
            state = state.unsqueeze(0).to(self.device)
            policy_logits, _ = self.model(state)
            # Remove batch dimension for the output
            policy_logits = policy_logits.squeeze(0)

            # Mask illegal actions
            legal_actions = chess_env.get_legal_actions(state.squeeze(0).cpu())
            mask = torch.full_like(policy_logits, float('-inf'))
            mask[legal_actions] = 0
            masked_logits = policy_logits + mask

            # Select action with the highest logit
            action_idx = torch.argmax(masked_logits).item()
            logger.debug(f"Agent selected action index: {action_idx} ({chess_env.action_index_to_uci(action_idx)})")
            return action_idx
