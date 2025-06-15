import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from env import ChessEnv
from utils import get_config


# Load corresponding config dictionary
config = get_config(__file__)


class Node:
    def __init__(self, state, sim_num=config['sim_num'], parent=None, use_variable_sim_num=config['use_variable_sim_num']):
        self.state = state
        self.parent = parent if parent is not None else {} # action -> parent node
        self.children = {} # action -> child node
        self.visits = 0
        self.total_reward = 0
        self.sim_num = sim_num
        self.use_variable_sim_num = use_variable_sim_num
        self.untried_actions = ChessEnv.get_legal_actions(state)

    def if_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def select_child_ucb(self):
        def _ucb(child):
            q = child.total_reward / (child.visits + 1e-6)
            u = config['c_puct'] * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            return q + u

        return max(self.children.items(), key=lambda x: _ucb(x[1]))


class MCTS:
    def __init__(self, agent, sim_num=config['sim_num'], c_puct=config['c_puct'], value_coef=config['value_coef']):
        self.agent = agent
        self.node_dict = {} # state -> node
                
    def select_action(self, state):
        # This line should be redundant, yet kept in case later code makes a mistake
        root_node, type = self.get_or_create_node(state)
        print(f"MCTS: The node is {type}, this a mistake.") if type == 'created' else None

        num_sims_to_run = self.sim_num

        for _ in range(num_sims_to_run):
            self.sesb(root_node)
        
        if not root_node.children:
            raise ValueError("select_action:No children found for the root node")
        
        best_action = max(root_node.children.items(), key=lambda x: x[1].visits)[0]
        print(f"MCTS: Best action: {best_action}, visits: {root_node.children[best_action].visits}")
        return best_action

    def sesb(self, root_node):
        node = root_node
        path = [node]

        # Selection following UCB
        while node.is_fully_expanded() and node.children:
            action, node = node.select_child_ucb()
            path.append(node)
        
        # Expansion
        if node.untried_actions:
            action = node.untried_actions.pop()
            next_state, reward, terminated, truncated, info = ChessEnv.step(node.state, action)
            child = self.get_or_create_node(next_state, parent=node, action=action)
            node.children[action] = child
            node = child
            path.append(node)
        
        # Simulation
        reward = self.simulate_with_agent(node.state, self.agent)

        # Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.total_reward += reward
            reward = -reward

    def get_or_create_node(self, state, parent=None, action=None):
        if state in self.node_dict:
            node = self.node_dict[state]
            if parent and action is not None:
                node.parent[action] = parent
            return node, 'got'
        else:
            node = Node(state.clone())
            if parent and action is not None:
                node.parent[action] = parent
            self.node_dict[state] = node
            return node, 'created'

    def is_fully_expanded(self, node):
        return len(node.untried_actions) == 0

    def simulate_with_agent(self, state, agent):
        state = state.clone()
        terminated = False
        step = 0
        max_depth = 100
        
        while not terminated and step < max_depth:
            step += 1
            # Get legal actions for the current state
            legal_actions = ChessEnv.get_legal_actions(state)
            if not legal_actions:
                raise ValueError("Checkmate or stalemate undetected by the environment")
            with torch.no_grad():
                policy_logits = agent(state.unsqueeze(0))[0].squeeze(0)  # Get logits for this state
                mask = torch.zeros_like(policy_logits, dtype=torch.bool)
                mask[legal_actions] = True  # Mark legal actions
                masked_logits = policy_logits.masked_fill(~mask, float('-inf'))  # Mask illegal actions
                probs = torch.softmax(masked_logits, dim=0)  # Probabilities over legal actions
                action = torch.multinomial(probs, 1).item()  # Sample an action
            # Step the environment
            next_state, reward, terminated, truncated, info = ChessEnv.step(state, action)
            state = next_state  # Update state
        return reward  # Return the final reward


def train_mcts(agent, optimizer, mcts_dataset, value_coef=config['value_coef'], weight_decay=1e-4, epochs=config['epochs'], batch_size=config['batch_size']):
    loader = DataLoader(mcts_dataset, batch_size=batch_size, shuffle=True)
    agent.train()

    for epoch in range(epochs):
        for states, pi_targets, z_targets in loader:
            policy_logits, values = agent(states)  # (B, 4672), (B, 1)

            policy_loss = F.cross_entropy(policy_logits, pi_targets.argmax(dim=1))  # pi_targets: (B, 4672)
            value_loss = F.mse_loss(values.squeeze(), z_targets)  # z_targets: (B,)

            l2_penalty = sum((p**2).sum() for p in agent.parameters())
            loss = policy_loss + value_coef * value_loss + weight_decay * l2_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
