import torch
import torch.optim as optim
import torch.nn.functional as F

from memory import PPOMemory
from networks import ActorCriticNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    def __init__(self, state_size, action_size, action_bounds, lr=0.0003, gamma=0.99, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        self.actor_critic = ActorCriticNetwork(state_size, action_size, action_bounds).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = PPOMemory(batch_size)
        
    def select_action(self, state, evaluate=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # shape [1, state_size]

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state)

        # Return a scalar value if action is shape [1, 1] or [1]
        return action.squeeze(0), log_prob, value

    
    def store_transition(self, state, action, log_prob, value, reward, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state], dtype=torch.float32).to(device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([[action]], dtype=torch.float32).to(device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32).to(device)
            
        self.memory.store_memory(state, action, log_prob, value, reward, done)
        
    def learn(self):
        for _ in range(self.n_epochs):
            batches, states, actions, old_log_probs, old_values, rewards, dones = self.memory.generate_batches()
            
            advantages = torch.zeros_like(torch.tensor(rewards, dtype=torch.float32)).to(device)
            
            # Calculate advantages using GAE
            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * old_values[k+1] * (1-dones[k]) - old_values[k])
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t
                
            values = torch.tensor(old_values, dtype=torch.float32).to(device)
            
            for batch in batches:
                states_batch = torch.cat([states[i] for i in batch]).to(device)
                actions_batch = torch.cat([actions[i] for i in batch]).to(device)
                old_log_probs_batch = torch.tensor([old_log_probs[i] for i in batch], dtype=torch.float32).to(device)
                advantages_batch = advantages[batch]
                
                # Get new log probs and entropy
                new_log_probs, entropy, critic_value = self.actor_critic.evaluate_actions(states_batch, actions_batch)
                
                # Compute ratio (π_θ / π_θold)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Compute surrogate losses
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages_batch
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                
                returns = advantages_batch + values[batch]
                critic_loss = F.mse_loss(critic_value.squeeze(-1), returns)
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # Perform optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
        # Clear memory after learning
        self.memory.clear_memory()