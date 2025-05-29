from env import IterativeAdAuctionEnv

class MultiAgentIterativeAdAuctionEnv:
    """
    A wrapper to manage multiple agents in the iterative ad auction environment.
    Creates a separate Gymnasium environment for each agent.
    """
    def __init__(self, num_agents=3, num_ad_slots=2, num_user_types=3, max_rounds=1000, 
                max_iterations_per_round=5, min_bid_increment=0.1):
        """
        Initialize the multi-agent environment
        
        Args:
            num_agents: Number of agents in the auction
            num_ad_slots: Number of available ad slots
            num_user_types: Number of different user types
            max_rounds: Maximum number of auction rounds
            max_iterations_per_round: Maximum number of bid iterations per round
            min_bid_increment: Minimum bid increment to consider bid as changed
        """
        self.num_agents = num_agents
        self.num_ad_slots = num_ad_slots
        self.num_user_types = num_user_types
        self.max_rounds = max_rounds
        self.max_iterations_per_round = max_iterations_per_round
        self.min_bid_increment = min_bid_increment
        
        # Create a separate environment for each agent
        self.envs = []
        for i in range(num_agents):
            env = IterativeAdAuctionEnv(
                num_agents=num_agents,
                num_ad_slots=num_ad_slots,
                num_user_types=num_user_types,
                max_rounds=max_rounds,
                agent_id=i,
                max_iterations_per_round=max_iterations_per_round,
                min_bid_increment=min_bid_increment
            )
            self.envs.append(env)
        
        # Track the current state of each agent
        self.observations = [None] * num_agents
        self.rewards = [0] * num_agents
        self.terminateds = [False] * num_agents
        self.truncateds = [False] * num_agents
        self.infos = [{}] * num_agents
        
        # Current round and iteration tracking
        self.current_round = 0
        self.current_iteration = 0
        self.bidding_finished = False
        
    def reset(self, seed=None):
        """
        Reset all environments
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observations: List of observations for each agent
            infos: List of info dictionaries for each agent
        """
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed)
            observations.append(obs)
            infos.append(info)
        
        self.observations = observations
        self.infos = infos
        
        # Reset round and iteration trackers
        self.current_round = 0
        self.current_iteration = 0
        self.bidding_finished = False
        
        return observations, infos
    
    def step(self, actions):
        """
        Take a step in all environments with the given actions
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            terminateds: List of terminated flags for each agent
            truncateds: List of truncated flags for each agent
            infos: List of info dicts for each agent
            episode_terminated: Flag indicating if any agent has terminated
            episode_truncated: Flag indicating if any agent has truncated
        """
        # Set opponent actions in each environment
        for i, env in enumerate(self.envs):
            opponent_actions = []
            for j, action in enumerate(actions):
                if j != i:
                    opponent_actions.append(action)
            env.set_opponent_actions(actions)
        
        # Take a step in each environment
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        # Update the current state
        self.observations = observations
        self.rewards = rewards
        self.terminateds = terminateds
        self.truncateds = truncateds
        self.infos = infos
        
        # Update bidding status
        if any([info['bidding_finished'] for info in infos]):
            self.bidding_finished = True
            self.current_iteration = 0
            self.current_round += 1
        else:
            self.current_iteration += 1
        
        # Check if any agent has terminated or truncated
        episode_terminated = all(terminateds)
        episode_truncated = any(truncateds)
        
        return observations, rewards, terminateds, truncateds, infos, episode_terminated, episode_truncated
    
    def get_current_status(self):
        """
        Get the current status of the auction environment
        
        Returns:
            Dictionary containing current round, iteration, bids, winners, etc.
        """
        return {
            'round': self.current_round,
            'iteration': self.current_iteration,
            'bidding_finished': self.bidding_finished,
            'current_bids': [env.current_bids[env.agent_id] for env in self.envs],
            'winners': self.envs[0].current_winners,  # Same for all envs
            'prices': self.envs[0].current_prices,    # Same for all envs
            'budgets': [env.budgets[env.agent_id] for env in self.envs],
            'clicks': [env.clicks[env.agent_id] for env in self.envs],
            'spend': [env.spend[env.agent_id] for env in self.envs],
            'revenue': [env.revenue[env.agent_id] for env in self.envs],
            'roi': [env.roi[env.agent_id] for env in self.envs]
        }
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()