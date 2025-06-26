import torch
import learning.ppo as ppo
import logging
import env


logger = logging.getLogger(__name__)


# Class name MUST match the agent name for importlib to work
class PpoResnet:
    def __init__(self, agent_config):
        logger.info(f"Initializing PpoResnet agent with config: {agent_config}")
        self.config = agent_config
        # Model config
        model_config = self.config['ppo']['model']
        self.model = ppo.ResNet(model_config=model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"PpoResnet agent model moved to device: {self.device}")

    def select_action(self, state):
        logger.debug("PpoResnet selecting action for given state tensor.")
        self.model.eval()

        legal_actions = env.get_legal_actions(state)

        state_tensor = state.unsqueeze(0).to(self.device)
        policy_logits, _ = self.model(state_tensor)
        policy_logits = policy_logits.squeeze(0)

        # Mask illegal actions
        mask = torch.full_like(policy_logits, float('-1e9'))
        mask[legal_actions] = 0
        
        policy_logits = policy_logits + mask
        probs = torch.softmax(policy_logits, dim=0)
        action = torch.multinomial(probs, 1).item()
        logger.debug(f"Agent selected action index: {action}")
        return action
