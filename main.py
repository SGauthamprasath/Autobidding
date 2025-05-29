# streamlit_app.py
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import numpy as np
import torch
import pandas as pd

from multi_env import MultiAgentIterativeAdAuctionEnv
from agent import PPOAgent
from utils import (
    train_agents,
    evaluate_agents,
    plot_training_results,
    plot_smoothed_training_results,
    save_agents,
    load_agents,
    print_evaluation_results
)

st.set_page_config(page_title="Multi-Agent Bidding Simulator", layout="wide")

st.title("🧠 Multi-Agent Iterative Ad Auction Simulator")
st.write("This app simulates agents bidding in iterative ad auctions using PPO.")

# Sidebar - Environment settings
st.sidebar.header("🔧 Environment Configuration")
num_agents = st.sidebar.slider("Number of agents", 2, 5, 3)
num_ad_slots = st.sidebar.slider("Ad slots", 1, 3, 2)
num_user_types = st.sidebar.slider("User types", 1, 5, 3)
max_rounds = st.sidebar.slider("Rounds per episode", 5, 100, 10)
max_iterations = st.sidebar.slider("Iterations per round", 1, 10, 3)
min_bid_increment = st.sidebar.slider("Min bid increment", 0.01, 0.5, 0.1)

# Initialize session state
if "env" not in st.session_state:
    st.session_state.env = None
if "agents" not in st.session_state:
    st.session_state.agents = None
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize environment
def init_environment():
    env = MultiAgentIterativeAdAuctionEnv(
        num_agents=num_agents,
        num_ad_slots=num_ad_slots,
        num_user_types=num_user_types,
        max_rounds=max_rounds,
        max_iterations_per_round=max_iterations,
        min_bid_increment=min_bid_increment
    )
    agents = []
    state_size = env.envs[0].state_size
    action_size = 1
    action_bounds = (env.envs[0].min_bid, env.envs[0].max_bid)

    for _ in range(num_agents):
        agent = PPOAgent(state_size, action_size, action_bounds, lr=0.001)
        agents.append(agent)

    try:
        load_agents(agents, path_prefix="iterative_ppo_agent")
        st.success("✅ Pretrained agents loaded")
    except:
        st.warning("⚠️ Using randomly initialized agents")

    st.session_state.env = env
    st.session_state.agents = agents
    st.session_state.history = []

# Step through one episode
def step():
    env = st.session_state.env
    agents = st.session_state.agents

    obs, _ = env.reset()
    done = False
    episode_history = []

    while not done:
        actions = [agent.select_action(o)[0].squeeze().item() for agent, o in zip(agents, obs)]

        obs, rewards, terminateds, truncateds, infos, episode_done, _ = env.step(actions)

        for i in range(num_agents):
            episode_history.append({
                "Agent": f"Agent {i+1}",
                "Bid": actions[i],
                "Reward": rewards[i],
                "Winning": infos[i]['is_winner']
            })

        if episode_done:
            done = True

    st.session_state.history.extend(episode_history)

# Buttons
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔁 Initialize Environment"):
        init_environment()
    if st.button("▶️ Run Auction Episode"):
        if st.session_state.env is not None:
            step()
        else:
            st.warning("Please initialize the environment first.")

# Display Results
if st.session_state.history:
    st.subheader("📊 Auction Results Log")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    reward_chart = hist_df.groupby("Agent")["Reward"].sum().reset_index()
    st.bar_chart(reward_chart.set_index("Agent"))

    bid_chart = hist_df.groupby("Agent")["Bid"].mean().reset_index()
    st.line_chart(bid_chart.set_index("Agent"))
