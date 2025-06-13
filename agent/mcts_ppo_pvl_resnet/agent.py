import torch
import torch.optim as optim
import learning.ppo as ppo
import os


# Class name !!!MUST!!! match the agent name for importlib to work
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
        self.config = {}

# Selection may not work as it is not masked
    def select_action(self, state):
        self.model.eval()
        with torch.no_grad():
            policy_logits, _ = self.model(state)
            action_idx = torch.argmax(policy_logits, dim=1).item() 
            return action_idx

    def save(self, path):
        torch.save({
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(path, 'checkpoint.pth'))

    def load(self, path):
        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.config = checkpoint.get('config', {})
