import numpy as np
import gymnasium as gym
from gymnasium import spaces

class IterativeAdAuctionEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, num_agents=3, num_ad_slots=2, num_user_types=3, max_rounds=1000, 
                 agent_id=0, render_mode=None, max_iterations_per_round=5, min_bid_increment=0.1):
        
        super(IterativeAdAuctionEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_ad_slots = num_ad_slots
        self.num_user_types = num_user_types
        self.max_rounds = max_rounds
        self.agent_id = agent_id  # The agent this environment is for
        self.render_mode = render_mode
        
        # Iterative bidding parameters
        self.max_iterations_per_round = max_iterations_per_round
        self.min_bid_increment = min_bid_increment
        self.current_iteration = 0
        self.bidding_finished = False
        self.all_bids_final = [False] * num_agents  # Track if each agent's bid is final
        
        # Initial budgets for each agent (starting budget)
        self.initial_budget = 10000
        self.budgets = [self.initial_budget] * num_agents
        
        # User type characteristics (CTR base rates per user type)
        self.user_ctr_base = np.random.uniform(0.01, 0.05, num_user_types)
        
        # Agent-specific CTR modifiers for each user type (representing targeting quality)
        self.agent_ctr_modifiers = np.random.uniform(0.8, 1.2, (num_agents, num_user_types))
        
        # Value per click for each agent (representing conversion value)
        self.value_per_click = np.random.uniform(1.0, 5.0, num_agents)
        
        # Quality scores for each agent (ad relevance)
        self.quality_scores = np.random.uniform(0.5, 1.0, num_agents)
        
        # Position-based CTR modifiers for ad slots
        self.position_ctr_modifiers = np.linspace(1.0, 0.5, num_ad_slots)
        
        # Additional state features for iterative bidding
        self.state_size = 1 + 1 + 1 + 1 + 1 + 1 + (num_agents - 1) + 2  # Added iteration and current winner information
        
        # Define action and observation space
        # Action: Continuous bid amount between min_bid and max_bid
        self.min_bid = 0.0
        self.max_bid = 5.0
        self.action_space = spaces.Box(
            low=np.array([self.min_bid]),
            high=np.array([self.max_bid]),
            dtype=np.float32
        )
        
        # Observation: State components
        obs_low = np.zeros(self.state_size)
        obs_high = np.ones(self.state_size)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Current simulation state
        self.current_round = 0
        self.current_user_type = 0
        
        # Historical performance tracking
        self.historical_ctr = np.zeros(num_agents)
        self.historical_cpc = np.zeros(num_agents)
        self.impressions = np.zeros(num_agents)
        self.clicks = np.zeros(num_agents)
        self.spend = np.zeros(num_agents)
        self.revenue = np.zeros(num_agents)
        self.roi = np.zeros(num_agents)
        
        # Current bids and previous bids from each agent
        self.current_bids = np.zeros(num_agents)
        self.previous_bids = np.zeros(num_agents)
        
        # Winning slots information (which agent won which slot)
        self.current_winners = [-1] * num_ad_slots
        self.current_prices = [0.0] * num_ad_slots
        
        # Opponent actions (will be set externally)
        self.opponent_actions = [0.0] * (self.num_agents - 1)
        
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        self.current_round = 0
        self.budgets = [self.initial_budget] * self.num_agents
        
        # Reset historical metrics
        self.historical_ctr = np.zeros(self.num_agents)
        self.historical_cpc = np.zeros(self.num_agents)
        self.impressions = np.zeros(self.num_agents)
        self.clicks = np.zeros(self.num_agents)
        self.spend = np.zeros(self.num_agents)
        self.revenue = np.zeros(self.num_agents)
        self.roi = np.zeros(self.num_agents)
        
        # Reset bidding information
        self.current_bids = np.zeros(self.num_agents)
        self.previous_bids = np.zeros(self.num_agents)
        self.current_iteration = 0
        self.bidding_finished = False
        self.all_bids_final = [False] * self.num_agents
        
        # Generate random user type for first round
        self.current_user_type = np.random.randint(0, self.num_user_types)
        
        # Create initial state for the agent
        observation = self._get_state()
        info = {}
        
        return observation, info
        
    def _get_state(self):
        """Get the state representation for the agent"""
        agent_id = self.agent_id
        
        # Budget normalized to [0,1]
        normalized_budget = self.budgets[agent_id] / self.initial_budget
        
        # Remaining auction rounds normalized to [0,1]
        remaining_rounds = (self.max_rounds - self.current_round) / self.max_rounds
        
        # Current user type (normalized)
        user_type = self.current_user_type / self.num_user_types
        
        # Historical CTR
        historical_ctr = self.historical_ctr[agent_id]
        
        # Historical cost per click
        historical_cpc = self.historical_cpc[agent_id] / 5.0  # Normalized assuming max CPC is 5.0
        
        # Quality score
        quality_score = self.quality_scores[agent_id]
        
        # Competitor bids (from current iteration)
        competitor_bids = []
        for i in range(self.num_agents):
            if i != agent_id:
                competitor_bids.append(self.current_bids[i] / 5.0)  # Normalized assuming max bid is 5.0
        
        # Current bidding iteration (normalized)
        normalized_iteration = self.current_iteration / self.max_iterations_per_round
        
        # Am I currently winning a slot? (1 if yes, 0 if no)
        is_winning = 1.0 if agent_id in self.current_winners else 0.0
        
        # Combine all state components
        state_components = [normalized_budget, remaining_rounds, user_type, 
                           historical_ctr, historical_cpc, quality_score] + competitor_bids + [normalized_iteration, is_winning]
        
        return np.array(state_components, dtype=np.float32)
    
    def set_opponent_actions(self, actions):
        """Set actions for opponent agents"""
        action_index = 0
        for i in range(self.num_agents):
            if i != self.agent_id:
                self.opponent_actions[action_index] = actions[i]
                action_index += 1
    
    def step(self, action):
        # Extract the bid amount from the action
        if isinstance(action, np.ndarray):
            bid = action[0]
        else:
            bid = action
            
        # Ensure bid is within bounds
        bid = max(self.min_bid, min(self.max_bid, bid))
        
        # Create full actions array with all agents
        all_actions = np.zeros(self.num_agents)
        action_index = 0
        for i in range(self.num_agents):
            if i == self.agent_id:
                all_actions[i] = bid
            else:
                all_actions[i] = self.opponent_actions[action_index]
                action_index += 1
        
        # Store bids
        self.previous_bids = self.current_bids.copy()
        self.current_bids = all_actions
        
        # Updated algorithm for determining when bids are final
        for i in range(self.num_agents):
            # The bid is final if:
            # 1. Current bid is within min_bid_increment of previous bid (agent is not increasing significantly)
            # 2. Or we've reached the maximum iterations
            if ((self.current_bids[i] - self.previous_bids[i]) < self.min_bid_increment) or \
               (self.current_iteration >= self.max_iterations_per_round - 1):
                self.all_bids_final[i] = True
        
        # Calculate effective bids (bid * quality score)
        effective_bids = all_actions * self.quality_scores
        
        # Get winning agents for each ad slot (second price auction)
        sorted_indices = np.argsort(-effective_bids)
        self.current_winners = sorted_indices[:self.num_ad_slots].tolist()
        
        # Calculate payments (second price auction - but only if bid is final)
        self.current_prices = [0.0] * self.num_ad_slots
        for slot, winner_idx in enumerate(self.current_winners):
            if slot + 1 < len(sorted_indices):  # If there's a next highest bidder
                next_highest_idx = sorted_indices[slot + 1]
                # Payment is next highest effective bid divided by winner's quality score
                self.current_prices[slot] = (effective_bids[next_highest_idx] / self.quality_scores[winner_idx])
            else:
                # If no next highest bid, use a reserve price (e.g., 0.1)
                self.current_prices[slot] = 0.1
        
        # Check if bidding is finished for this auction round
        self.current_iteration += 1
        
        if self.current_iteration >= self.max_iterations_per_round or all(self.all_bids_final):
            self.bidding_finished = True
        
        # Determine rewards and state transitions
        if self.bidding_finished:
            # Process clicks and payments only when bidding is finished
            clicks = np.zeros(self.num_agents)
            revenue = np.zeros(self.num_agents)
            
            # Process each ad slot
            for slot, winner_idx in enumerate(self.current_winners):
                # Base CTR for the current user type
                base_ctr = self.user_ctr_base[self.current_user_type]
                
                # Modify CTR based on agent-specific targeting quality
                agent_ctr_modifier = self.agent_ctr_modifiers[winner_idx, self.current_user_type]
                
                # Modify CTR based on ad position
                position_ctr_modifier = self.position_ctr_modifiers[slot]
                
                # Final CTR
                final_ctr = base_ctr * agent_ctr_modifier * position_ctr_modifier
                
                # Calculate actual clicks (stochastic)
                actual_clicks = np.random.binomial(1, final_ctr)
                
                # Update agent metrics
                clicks[winner_idx] += actual_clicks
                revenue[winner_idx] += actual_clicks * self.value_per_click[winner_idx]
                
                # Deduct payment from budget if there was a click
                if actual_clicks > 0:
                    payment = self.current_prices[slot]
                    self.budgets[winner_idx] -= payment
                    self.spend[winner_idx] += payment
            
            # Update historical performance metrics
            for i in range(self.num_agents):
                if i in self.current_winners:
                    self.impressions[i] += 1
                    self.clicks[i] += clicks[i]
                    
                    # Update historical CTR
                    if self.impressions[i] > 0:
                        self.historical_ctr[i] = self.clicks[i] / self.impressions[i]
                    
                    # Update historical CPC
                    if self.clicks[i] > 0:
                        self.historical_cpc[i] = self.spend[i] / self.clicks[i]
                    
                    # Update revenue
                    self.revenue[i] += revenue[i]
                    
                    # Update ROI
                    if self.spend[i] > 0:
                        self.roi[i] = (self.revenue[i] - self.spend[i]) / self.spend[i]
            
            # Move to next round
            self.current_round += 1
            
            # Generate random user type for next round
            self.current_user_type = np.random.randint(0, self.num_user_types)
            
            # Reset bidding state for next round
            self.current_iteration = 0
            self.bidding_finished = False
            self.all_bids_final = [False] * self.num_agents
            
            # Calculate reward for the agent
            agent_id = self.agent_id
            
            # 1. Immediate reward from revenue gained minus cost
            immediate_reward = revenue[agent_id] - (clicks[agent_id] * self.current_prices[self.current_winners.index(agent_id)] if agent_id in self.current_winners else 0)
            
            # 2. Penalize if budget is depleted too early
            budget_depletion_penalty = -10.0 if (self.budgets[agent_id] <= 0 and self.current_round < self.max_rounds * 0.8) else 0.0
            
            # 3. Reward for efficient budget utilization at the end of episode
            budget_efficiency_reward = 0.0
            if self.current_round >= self.max_rounds or self.budgets[agent_id] <= 0:
                # Reward for using budget effectively (close to 0 remaining is good)
                # but penalize for having a lot of unused budget
                budget_efficiency = 1.0 - (self.budgets[agent_id] / self.initial_budget)
                budget_efficiency_reward = 5.0 * budget_efficiency
                
                # Add final ROI bonus
                if self.spend[agent_id] > 0:
                    roi_bonus = 10.0 * self.roi[agent_id]
                    budget_efficiency_reward += roi_bonus
            
            # Combine reward components
            reward = immediate_reward + budget_depletion_penalty + budget_efficiency_reward
            
        else:
            # During iterative bidding, give small rewards for strategic bidding
            agent_id = self.agent_id
            
            # Small reward if agent increased their bid to potentially win a slot
            # But only if they're not already winning or their current bid wasn't high enough
            if agent_id not in self.current_winners:
                # If agent increased bid - small positive reward
                if self.current_bids[agent_id] > self.previous_bids[agent_id]:
                    reward = 0.05
                else:
                    reward = -0.01  # Small penalty for not increasing bid when not winning
            else:
                # If agent is winning and stays competitive - small positive reward
                reward = 0.02
        
        # Check if episode is terminated or truncated
        terminated = False
        truncated = False
        
        if self.current_round >= self.max_rounds:
            truncated = True
        
        # Also mark as terminated if budget is depleted
        if self.budgets[self.agent_id] <= 0:
            terminated = True
            self.budgets[self.agent_id] = 0  # Ensure budget doesn't go negative
        
        # Get next state
        observation = self._get_state()
        
        # Additional info
        info = {
            'iteration': self.current_iteration,
            'bidding_finished': self.bidding_finished,
            'is_winner': self.agent_id in self.current_winners,
            'clicks': self.clicks[self.agent_id],
            'revenue': self.revenue[self.agent_id],
            'spend': self.spend[self.agent_id],
            'roi': self.roi[self.agent_id],
            'budget_remaining': self.budgets[self.agent_id]
        }
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            agent_id = self.agent_id
            print(f"Round: {self.current_round}/{self.max_rounds}, Iteration: {self.current_iteration}")
            print(f"Agent {agent_id} Budget: ${self.budgets[agent_id]:.2f}")
            print(f"Agent {agent_id} Bid: ${self.current_bids[agent_id]:.2f}")
            print(f"Current Winners: {self.current_winners}")
            print(f"Current Prices: {[f'${p:.2f}' for p in self.current_prices]}")
            print(f"Winning: {'Yes' if agent_id in self.current_winners else 'No'}")
            print("-" * 40)
    
    def close(self):
        """Close the environment"""
        pass