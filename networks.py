import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_bounds):
        super(ActorCriticNetwork, self).__init__()
        self.action_size = action_size
        self.action_bounds = action_bounds  # (min_bid, max_bid)
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, 128)
        self.shared_fc2 = nn.Linear(128, 128)
        
        # Actor (policy) network
        self.actor_fc1 = nn.Linear(128, 64)
        self.actor_mean = nn.Linear(64, action_size)
        self.actor_log_std = nn.Linear(64, action_size)
        
        # Critic (value) network
        self.critic_fc1 = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, state):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor
        ax = F.relu(self.actor_fc1(x))
        action_mean = torch.tanh(self.actor_mean(ax))  # Tanh to bound between -1 and 1
        action_log_std = self.actor_log_std(ax)
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)  # Prevent numerical instability
        action_std = torch.exp(action_log_std)
        
        # Scale mean from [-1, 1] to action bounds
        action_low, action_high = self.action_bounds
        action_mean = ((action_mean + 1) / 2) * (action_high - action_low) + action_low
        
        # Critic
        cx = F.relu(self.critic_fc1(x))
        value = self.critic(cx)
        
        return action_mean, action_std, value
    
    def get_action(self, state, evaluate=False):
        action_mean, action_std, value = self.forward(state)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        if evaluate:
            action = action_mean
        else:
            action = dist.sample()
            
        # Ensure action is within bounds
        action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions):
        action_mean, action_std, value = self.forward(states)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        
        return log_probs, entropy, value