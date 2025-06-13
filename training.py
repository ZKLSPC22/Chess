import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


class PPO_train:
    def __init__(self, agent, optimizer, transition_dataset, clip_eps, entropy_coef, value_coef, epochs, batch_size):
        self.agent = agent
        self.optimizer = optimizer
        self.transition_dataset = transition_dataset
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.loader = DataLoader(transition_dataset, batch_size=batch_size, shuffle=True)
        self.policy = set('offline', 'offpolicy', 'nn')

    def train(self):
        for epoch in range(self.epochs):
            for states, actions, old_log_probs, returns, advantages in self.loader:
                policy_logits, values = self.agent(states)
            # Apply softmax to policy logits to get probabilities
            dist = torch.distributions.Categorical(logits=policy_logits)
            # Recalculate log probabilities to normalize the policy
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


