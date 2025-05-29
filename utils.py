import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_agents(multi_env, agents, num_episodes=100, max_steps_per_episode=10000):
    """
    Train multiple PPO agents in the iterative auction environment
    
    Args:
        multi_env: MultiAgentIterativeAdAuctionEnv instance
        agents: List of PPOAgent instances
        num_episodes: Number of episodes to train for
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        training_stats: Dictionary containing training statistics
    """
    # Training statistics
    stats = {
        'episode_rewards': [[] for _ in range(len(agents))],
        'episode_roi': [[] for _ in range(len(agents))],
        'episode_spend': [[] for _ in range(len(agents))],
        'episode_revenue': [[] for _ in range(len(agents))],
        'episode_clicks': [[] for _ in range(len(agents))],
        'episode_win_rate': [[] for _ in range(len(agents))],
    }
    
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # Reset environment and get initial states
        observations, _ = multi_env.reset(seed=episode)
        
        episode_rewards = [0] * len(agents)
        step_count = 0
        win_counts = [0] * len(agents)
        step_counts = [0] * len(agents)
        
        # Run episode
        while step_count < max_steps_per_episode:
            step_count += 1
            
            # Select actions for each agent
            actions = []
            log_probs = []
            values = []
            
            for i, agent in enumerate(agents):
                action, log_prob, value = agent.select_action(observations[i])
                actions.append(action.item())
                log_probs.append(log_prob)
                values.append(value)
            
            # Take a step in the environment
            next_observations, rewards, terminateds, truncateds, infos, episode_terminated, episode_truncated = multi_env.step(actions)
            
            # Store transitions for each agent
            for i, agent in enumerate(agents):
                agent.store_transition(
                    observations[i], 
                    actions[i], 
                    log_probs[i], 
                    values[i], 
                    rewards[i], 
                    terminateds[i] or truncateds[i]
                )
                
                episode_rewards[i] += rewards[i]
                
                # Track win rate
                if infos[i]['is_winner']:
                    win_counts[i] += 1
                step_counts[i] += 1
            
            # Update observations
            observations = next_observations
            
            # Check if bidding phase is complete for all agents
            if multi_env.bidding_finished:
                # Learn from experiences when the bidding round is complete
                for agent in agents:
                    agent.learn()
                    
                multi_env.bidding_finished = False
            
            # Break if episode is terminated or truncated
            if episode_terminated or episode_truncated:
                break
        
        # Record episode statistics
        status = multi_env.get_current_status()
        for i in range(len(agents)):
            stats['episode_rewards'][i].append(episode_rewards[i])
            stats['episode_roi'][i].append(status['roi'][i])
            stats['episode_spend'][i].append(status['spend'][i])
            stats['episode_revenue'][i].append(status['revenue'][i])
            stats['episode_clicks'][i].append(status['clicks'][i])
            stats['episode_win_rate'][i].append(win_counts[i] / max(1, step_counts[i]))
    
    return stats


def evaluate_agents(multi_env, agents, num_episodes=10, render=False):
    """
    Evaluate multiple trained PPO agents in the iterative auction environment
    
    Args:
        multi_env: MultiAgentIterativeAdAuctionEnv instance
        agents: List of PPOAgent instances
        num_episodes: Number of episodes to evaluate for
        render: Whether to render the environment
        
    Returns:
        eval_stats: Dictionary containing evaluation statistics
        avg_stats: Dictionary containing average evaluation metrics
    """
    # Evaluation statistics
    stats = {
        'episode_rewards': [[] for _ in range(len(agents))],
        'episode_roi': [[] for _ in range(len(agents))],
        'episode_spend': [[] for _ in range(len(agents))],
        'episode_revenue': [[] for _ in range(len(agents))],
        'episode_clicks': [[] for _ in range(len(agents))],
        'episode_win_rate': [[] for _ in range(len(agents))],
    }
    
    for episode in range(num_episodes):
        # Reset environment and get initial states
        observations, _ = multi_env.reset(seed=1000 + episode)  # Different seeds from training
        
        episode_rewards = [0] * len(agents)
        step_count = 0
        win_counts = [0] * len(agents)
        step_counts = [0] * len(agents)
        
        # Run episode
        while True:
            step_count += 1
            
            # Select actions for each agent (in evaluation mode)
            actions = []
            
            for i, agent in enumerate(agents):
                action, _, _ = agent.select_action(observations[i], evaluate=True)
                actions.append(action.item())
            
            # Take a step in the environment
            next_observations, rewards, terminateds, truncateds, infos, episode_terminated, episode_truncated = multi_env.step(actions)
            
            for i in range(len(agents)):
                episode_rewards[i] += rewards[i]
                
                # Track win rate
                if infos[i]['is_winner']:
                    win_counts[i] += 1
                step_counts[i] += 1
            
            # Render if requested
            if render and episode == 0:
                status = multi_env.get_current_status()
                print(f"Round: {status['round']}, Iteration: {status['iteration']}")
                print(f"Bids: {status['current_bids']}")
                print(f"Winners: {status['winners']}")
                print(f"Prices: {status['prices']}")
                print(f"Budgets: {status['budgets']}")
                print("-" * 40)
            
            # Update observations
            observations = next_observations
            
            # Break if episode is terminated or truncated
            if episode_terminated or episode_truncated:
                break
        
        # Record episode statistics
        status = multi_env.get_current_status()
        for i in range(len(agents)):
            stats['episode_rewards'][i].append(episode_rewards[i])
            stats['episode_roi'][i].append(status['roi'][i])
            stats['episode_spend'][i].append(status['spend'][i])
            stats['episode_revenue'][i].append(status['revenue'][i])
            stats['episode_clicks'][i].append(status['clicks'][i])
            stats['episode_win_rate'][i].append(win_counts[i] / max(1, step_counts[i]))
    
    # Calculate average metrics
    avg_stats = {
        'avg_reward': [np.mean(stats['episode_rewards'][i]) for i in range(len(agents))],
        'avg_roi': [np.mean(stats['episode_roi'][i]) for i in range(len(agents))],
        'avg_spend': [np.mean(stats['episode_spend'][i]) for i in range(len(agents))],
        'avg_revenue': [np.mean(stats['episode_revenue'][i]) for i in range(len(agents))],
        'avg_clicks': [np.mean(stats['episode_clicks'][i]) for i in range(len(agents))],
        'avg_win_rate': [np.mean(stats['episode_win_rate'][i]) for i in range(len(agents))],
    }
    
    return stats, avg_stats


def plot_training_results(stats, agent_names=None):
    """
    Plot training statistics
    
    Args:
        stats: Dictionary containing training statistics
        agent_names: List of agent names for the legend
    """
    if agent_names is None:
        agent_names = [f"Agent {i}" for i in range(len(stats['episode_rewards']))]
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot rewards
    for i in range(len(agent_names)):
        axs[0, 0].plot(stats['episode_rewards'][i], label=agent_names[i])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot ROI
    for i in range(len(agent_names)):
        axs[0, 1].plot(stats['episode_roi'][i], label=agent_names[i])
    axs[0, 1].set_title('Return on Investment (ROI)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('ROI')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot spend
    for i in range(len(agent_names)):
        axs[1, 0].plot(stats['episode_spend'][i], label=agent_names[i])
    axs[1, 0].set_title('Total Spend')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Spend')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot revenue
    for i in range(len(agent_names)):
        axs[1, 1].plot(stats['episode_revenue'][i], label=agent_names[i])
    axs[1, 1].set_title('Total Revenue')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Revenue')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot clicks
    for i in range(len(agent_names)):
        axs[2, 0].plot(stats['episode_clicks'][i], label=agent_names[i])
    axs[2, 0].set_title('Total Clicks')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Clicks')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)
    
    # Plot win rate
    for i in range(len(agent_names)):
        axs[2, 1].plot(stats['episode_win_rate'][i], label=agent_names[i])
    axs[2, 1].set_title('Win Rate')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Win Rate')
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_agents(agents, path_prefix="it_ppo_agent"):
    """
    Save trained agent models to disk
    
    Args:
        agents: List of PPOAgent instances
        path_prefix: Prefix for saved file names
    """
    for i, agent in enumerate(agents):
        path = f"{path_prefix}_agent{i+1}.pt"
        torch.save(agent.actor_critic.state_dict(), path)
        print(f"Saved agent {i+1} model to {path}")


def load_agents(agents, path_prefix="it_ppo_agent"):
    """
    Load trained agent models from disk
    
    Args:
        agents: List of PPOAgent instances
        path_prefix: Prefix for saved file names
    """
    for i, agent in enumerate(agents):
        path = f"{path_prefix}_agent{i+1}.pt"
        try:
            agent.actor_critic.load_state_dict(torch.load(path))
            print(f"Loaded agent {i+1} model from {path}")
        except FileNotFoundError:
            print(f"Warning: Could not find saved model at {path}")


def print_evaluation_results(avg_stats, agent_names=None, learning_rates=None):
    """
    Print formatted evaluation results
    
    Args:
        avg_stats: Dictionary containing average evaluation statistics
        agent_names: List of agent names
        learning_rates: List of learning rates for each agent
    """
    print("\nEvaluation Results:")
    print("=" * 60)
    
    for i in range(len(avg_stats['avg_reward'])):
        if agent_names:
            agent_name = agent_names[i]
        elif learning_rates:
            agent_name = f"Agent {i+1} (lr={learning_rates[i]})"
        else:
            agent_name = f"Agent {i+1}"
            
        print(f"\n{agent_name}:")
        print(f"  Average Reward: {avg_stats['avg_reward'][i]:.2f}")
        print(f"  Average ROI: {avg_stats['avg_roi'][i]:.2f}")
        print(f"  Average Spend: ${avg_stats['avg_spend'][i]:.2f}")
        print(f"  Average Revenue: ${avg_stats['avg_revenue'][i]:.2f}")
        print(f"  Average Clicks: {avg_stats['avg_clicks'][i]:.2f}")
        print(f"  Average Win Rate: {avg_stats['avg_win_rate'][i]:.2f}")


def calculate_moving_average(data, window_size=10):
    """
    Calculate moving average for smoothing plots
    
    Args:
        data: List of values
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data as numpy array
    """
    if len(data) < window_size:
        return np.array(data)
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return np.array(smoothed)


def plot_smoothed_training_results(stats, agent_names=None, window_size=10):
    """
    Plot smoothed training statistics using moving average
    
    Args:
        stats: Dictionary containing training statistics
        agent_names: List of agent names for the legend
        window_size: Size of moving average window for smoothing
    """
    if agent_names is None:
        agent_names = [f"Agent {i}" for i in range(len(stats['episode_rewards']))]
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot smoothed rewards
    for i in range(len(agent_names)):
        smoothed_rewards = calculate_moving_average(stats['episode_rewards'][i], window_size)
        axs[0, 0].plot(smoothed_rewards, label=agent_names[i], linewidth=2)
    axs[0, 0].set_title(f'Episode Rewards (Smoothed, window={window_size})')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot smoothed ROI
    for i in range(len(agent_names)):
        smoothed_roi = calculate_moving_average(stats['episode_roi'][i], window_size)
        axs[0, 1].plot(smoothed_roi, label=agent_names[i], linewidth=2)
    axs[0, 1].set_title(f'ROI (Smoothed, window={window_size})')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('ROI')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot smoothed spend
    for i in range(len(agent_names)):
        smoothed_spend = calculate_moving_average(stats['episode_spend'][i], window_size)
        axs[1, 0].plot(smoothed_spend, label=agent_names[i], linewidth=2)
    axs[1, 0].set_title(f'Total Spend (Smoothed, window={window_size})')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Spend')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot smoothed revenue
    for i in range(len(agent_names)):
        smoothed_revenue = calculate_moving_average(stats['episode_revenue'][i], window_size)
        axs[1, 1].plot(smoothed_revenue, label=agent_names[i], linewidth=2)
    axs[1, 1].set_title(f'Total Revenue (Smoothed, window={window_size})')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Revenue')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot smoothed clicks
    for i in range(len(agent_names)):
        smoothed_clicks = calculate_moving_average(stats['episode_clicks'][i], window_size)
        axs[2, 0].plot(smoothed_clicks, label=agent_names[i], linewidth=2)
    axs[2, 0].set_title(f'Total Clicks (Smoothed, window={window_size})')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Clicks')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)
    
    # Plot smoothed win rate
    for i in range(len(agent_names)):
        smoothed_win_rate = calculate_moving_average(stats['episode_win_rate'][i], window_size)
        axs[2, 1].plot(smoothed_win_rate, label=agent_names[i], linewidth=2)
    axs[2, 1].set_title(f'Win Rate (Smoothed, window={window_size})')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Win Rate')
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()