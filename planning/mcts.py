import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from env import ChessEnv


class Node:
    """A node in the MCTS tree."""
    def __init__(self, state, c_puct, parent=None, prior_p=0.0):
        self.state = state
        self.parent = parent if parent is not None else {} # action -> parent node
        self.children = {} # action -> child node
        self.visits = 0
        self.total_reward = 0
        self.c_puct = c_puct
        # The prior probability of selecting this node's action, from the parent's perspective.
        # This is obtained from the policy head of the neural network.
        self.prior_p = prior_p

    def select_child_ucb(self):
        """
        Selects a child node using the PUCT (Polynomial Upper Confidence for Trees) algorithm,
        as described in the AlphaZero paper. This balances exploiting moves with high average
        reward and exploring moves with high prior probability.
        """
        def _ucb(child):
            # Q-value: The average reward from the child's perspective.
            # We negate it because the child's reward is the parent's loss in a zero-sum game.
            q = -child.total_reward / (child.visits + 1e-6)
            
            # U-value: The exploration term.
            # Encourages exploration of actions with high prior probability and low visit counts.
            u = self.c_puct * child.prior_p * math.sqrt(self.visits) / (1 + child.visits)
            return q + u

        return max(self.children.items(), key=lambda x: _ucb(x[1]))


class MCTS:
    def __init__(self, agent, mcts_config):
        self.agent = agent
        self.num_simulations = mcts_config['num_simulations']
        self.c_puct = mcts_config['c_puct']
        self.value_coef = mcts_config['value_coef']
        self.node_dict = {} # state -> node
        self.root = None
                
    def _get_state_key(self, state):
        """Converts a state tensor to a hashable key (bytes)."""
        return state.tobytes()

    def reset(self):
        self.node_dict = {}
        self.root = None
                
    def select_action(self, state, chess_env, return_policy=False):
        if self.root is None or not torch.equal(self.root.state, state):
            self.root, type = self.get_or_create_node(state, chess_env)

        num_sims_to_run = self.num_simulations

        for _ in range(num_sims_to_run):
            self.sesb(self.root, chess_env)
        
        if not self.root.children:
            raise ValueError("select_action:No children found for the root node")
        
        # Create policy target
        policy_target = torch.zeros(4672, dtype=torch.float32)
        total_visits = 0
        for action, child in self.root.children.items():
            policy_target[action] = child.visits
            total_visits += child.visits
        
        if total_visits > 0:
            policy_target /= total_visits

        best_action = max(self.root.children.items(), key=lambda x: x[1].visits)[0]
        
        if return_policy:
            return best_action, policy_target
        return best_action

    def advance_tree(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            # Prune the tree, keeping only the subtree rooted at the new root.
            self.node_dict = {}
            
            q = [self.root]
            state_key = self._get_state_key(self.root.state)
            visited = {state_key}
            self.node_dict[state_key] = self.root
            self.root.parent = {}

            while q:
                node = q.pop(0)
                for child in node.children.values():
                    child_key = self._get_state_key(child.state)
                    if child_key not in visited:
                        self.node_dict[child_key] = child
                        visited.add(child_key)
                        q.append(child)
        else:
            self.root = None

    def sesb(self, root_node, chess_env):
        """
        Performs one simulation of the MCTS algorithm, consisting of three phases:
        1. Selection: Traverse the tree from the root by repeatedly selecting the child with the highest UCB value.
        2. Expansion & Evaluation: When a leaf node is reached, expand it by creating all its children.
           The leaf node is evaluated using the neural network, which provides a reward and prior probabilities for the new children.
        3. Backpropagation: Update the visit counts and total rewards of all nodes in the traversed path.
        """
        node = root_node
        path = [node]

        # Selection: Traverse the tree until a leaf node (a node with no children) is reached.
        while node.children:
            action, node = node.select_child_ucb()
            path.append(node)
        
        # Expansion and Evaluation: Once a leaf is found, expand it.
        legal_actions = chess_env.get_legal_actions(node.state)
        
        # Check if the game has ended at this node.
        if legal_actions:
            # If not a terminal node, use the network to get the policy and value.
            with torch.no_grad():
                # The model provides policy logits and a value from the current player's perspective.
                policy_logits, value = self.agent.model(node.state.unsqueeze(0).to(self.agent.device))
            
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            
            # Create a child node for each legal action.
            for action in legal_actions:
                # Note: The environment step is not needed here as we have the state representation.
                # This is a simplification assuming the state representation is sufficient.
                # A more rigorous implementation might step the env to get the child state.
                next_state, _, _, _, _ = chess_env.step(node.state, action)
                child, _ = self.get_or_create_node(
                    next_state,
                    c_puct=self.c_puct,
                    parent=node,
                    action=action,
                    prior_p=policy_probs[action].item()
                )
                node.children[action] = child
            
            # The reward for backpropagation is the value predicted by the network.
            reward = value.item()
        else:
            # If it's a terminal node, the reward is the actual game outcome.
            reward = chess_env._get_reward(chess_env._state_to_board(node.state)).item()

        # Backpropagation: Update the statistics of the nodes in the path.
        for n in reversed(path):
            n.visits += 1
            n.total_reward += reward
            # The reward must be inverted for the parent node.
            reward = -reward

    def get_or_create_node(self, state, chess_env=None, parent=None, action=None, c_puct=1.0, prior_p=0.0):
        # The byte representation of the state tensor is used as the key.
        state_key = self._get_state_key(state)
        if state_key in self.node_dict:
            node = self.node_dict[state_key]
            if parent and action is not None:
                node.parent[action] = parent
            return node, 'got'
        else:
            node = Node(state.clone(), c_puct=c_puct, prior_p=prior_p)
            if parent and action is not None:
                node.parent[action] = parent
            self.node_dict[state_key] = node
            return node, 'created'

def train_mcts(agent, optimizer, mcts_dataset, train_config):
    loader = DataLoader(mcts_dataset, batch_size=train_config['batch_size'], shuffle=True)
    agent.train()

    for epoch in range(train_config['epochs']):
        for states, pi_targets, z_targets in loader:
            policy_logits, values = agent(states)  # (B, 4672), (B, 1)

            policy_loss = F.cross_entropy(policy_logits, pi_targets.argmax(dim=1))  # pi_targets: (B, 4672)
            value_loss = F.mse_loss(values.squeeze(), z_targets)  # z_targets: (B,)

            l2_penalty = sum((p**2).sum() for p in agent.parameters())
            loss = policy_loss + train_config['value_coef'] * value_loss + train_config['weight_decay'] * l2_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
