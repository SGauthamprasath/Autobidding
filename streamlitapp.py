import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Ad Auction Platform",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .auction-stats {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .winner-card {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ActorCriticNetwork from ad_auction_gymnasium.py
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_bounds):
        super(ActorCriticNetwork, self).__init__()
        self.action_size = action_size
        self.action_bounds = action_bounds
        
        self.shared_fc1 = nn.Linear(state_size, 128)
        self.shared_fc2 = nn.Linear(128, 128)
        
        self.actor_fc1 = nn.Linear(128, 64)
        self.actor_mean = nn.Linear(64, action_size)
        self.actor_log_std = nn.Linear(64, action_size)
        
        self.critic_fc1 = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, state):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        ax = F.relu(self.actor_fc1(x))
        action_mean = torch.tanh(self.actor_mean(ax))
        action_log_std = self.actor_log_std(ax)
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)
        
        action_low, action_high = self.action_bounds
        action_mean = ((action_mean + 1) / 2) * (action_high - action_low) + action_low
        
        cx = F.relu(self.critic_fc1(x))
        value = self.critic(cx)
        
        return action_mean, action_std, value
    
    def get_action(self, state, evaluate=True):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = action_mean if evaluate else dist.sample()
        action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

# PPOAgent from ad_auction_gymnasium.py (simplified for inference)
class PPOAgent:
    def __init__(self, state_size, action_size, action_bounds):
        self.actor_critic = ActorCriticNetwork(state_size, action_size, action_bounds).to(device)
    
    def select_action(self, state, evaluate=True):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state], dtype=torch.float32).to(device)
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state, evaluate)
        return action.item()

# Initialize session state
def initialize_session_state():
    if 'auction_active' not in st.session_state:
        st.session_state.auction_active = False
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 1
    if 'current_iteration' not in st.session_state:
        st.session_state.current_iteration = 1
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    if 'auction_history' not in st.session_state:
        st.session_state.auction_history = []
    if 'round_history' not in st.session_state:
        st.session_state.round_history = []
    if 'current_bids' not in st.session_state:
        st.session_state.current_bids = {}
    if 'auction_config' not in st.session_state:
        st.session_state.auction_config = {
            'num_ad_slots': 2,
            'num_user_types': 3,
            'max_rounds': 10,
            'max_iterations_per_round': 5,
            'min_bid_increment': 0.1,
            'starting_budget': 100.0,
            'min_bid': 0.0,
            'max_bid': 5.0
        }
    if 'user_types' not in st.session_state:
        st.session_state.user_types = []
    if 'ppo_agents' not in st.session_state:
        st.session_state.ppo_agents = {}
    if 'previous_bids' not in st.session_state:
        st.session_state.previous_bids = {}

class Agent:
    def __init__(self, name: str, agent_type: str = "Human", budget: float = 100.0, 
                 model_path: Optional[str] = None, quality_score: Optional[float] = None):
        self.name = name
        self.agent_type = agent_type
        self.budget = budget
        self.initial_budget = budget
        self.total_spent = 0.0
        self.total_revenue = 0.0
        self.slots_won = 0
        self.bid_history = []
        self.performance_history = []
        self.color = self._generate_color()
        self.quality_score = quality_score if quality_score is not None else np.random.uniform(0.5, 1.0)
        self.ppo_agent = None
        if agent_type == "AI Bot" and model_path:
            self.ppo_agent = PPOAgent(state_size=8, action_size=1, action_bounds=(0.0, 5.0))
            self.ppo_agent.actor_critic.load_state_dict(torch.load(model_path, map_location=device))
            self.ppo_agent.actor_critic.eval()

    def _generate_color(self):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        return colors[hash(self.name) % len(colors)]
    
    def place_bid(self, amount: float) -> bool:
        if amount <= self.budget:
            self.bid_history.append(amount)
            return True
        return False
    
    def win_slot(self, bid_amount: float, revenue: float):
        self.budget -= bid_amount
        self.total_spent += bid_amount
        self.total_revenue += revenue
        self.slots_won += 1
    
    def get_roi(self) -> float:
        if self.total_spent == 0:
            return 0.0
        return (self.total_revenue - self.total_spent) / self.total_spent * 100
    
    def get_stats(self) -> Dict:
        return {
            'name': self.name,
            'type': self.agent_type,
            'budget': self.budget,
            'total_spent': self.total_spent,
            'total_revenue': self.total_revenue,
            'slots_won': self.slots_won,
            'roi': self.get_roi(),
            'avg_bid': np.mean(self.bid_history) if self.bid_history else 0.0,
            'bid_count': len(self.bid_history),
            'quality_score': self.quality_score
        }
    
    def get_ppo_bid(self, state: np.ndarray) -> float:
        if self.ppo_agent:
            bid = self.ppo_agent.select_action(state)
            return max(st.session_state.auction_config['min_bid_increment'], min(bid, self.budget))
        return 0.0

def create_auction_environment():
    config = st.session_state.auction_config
    user_types = []
    for i in range(config['num_user_types']):
        user_type = {
            'id': i,
            'name': f"User Type {i+1}",
            'click_rate': np.random.uniform(0.1, 0.3),
            'conversion_rate': np.random.uniform(0.05, 0.15),
            'value_per_conversion': np.random.uniform(5.0, 20.0)
        }
        user_types.append(user_type)
    return user_types

def get_auction_state(agent: Agent, user_type_id: int) -> np.ndarray:
    config = st.session_state.auction_config
    state = np.zeros(8, dtype=np.float32)
    state[0] = agent.budget / config['starting_budget']
    state[1] = (config['max_rounds'] - st.session_state.current_round) / config['max_rounds']
    state[2] = user_type_id / config['num_user_types']
    state[3] = 0.03
    state[4] = 0.5
    state[5] = agent.quality_score
    state[6] = np.mean(list(st.session_state.previous_bids.values()) + [0.0]) / config['max_bid'] if st.session_state.previous_bids else 0.5
    state[7] = np.max(list(st.session_state.previous_bids.values()) + [0.0]) / config['max_bid'] if st.session_state.previous_bids else 0.5
    return state

def calculate_auction_results(bids: Dict[str, float], user_types: List[Dict]) -> Dict:
    if not bids:
        return {'winners': [], 'payments': {}, 'revenues': {}}
    
    effective_bids = {}
    for agent_name, bid in bids.items():
        agent = st.session_state.agents[agent_name]
        effective_bids[agent_name] = bid * agent.quality_score
    
    sorted_effective_bids = sorted(effective_bids.items(), key=lambda x: x[1], reverse=True)
    num_slots = st.session_state.auction_config['num_ad_slots']
    winners = sorted_effective_bids[:num_slots]
    
    payments = {}
    revenues = {}
    slot_positions = {}
    
    for i, (agent_name, effective_bid) in enumerate(winners):
        if i + 1 < len(sorted_effective_bids):
            next_highest_effective_bid = sorted_effective_bids[i + 1][1]
            winner_quality_score = st.session_state.agents[agent_name].quality_score
            payment = next_highest_effective_bid / winner_quality_score
        else:
            payment = st.session_state.auction_config['min_bid_increment']
        
        original_bid = bids[agent_name]
        payments[agent_name] = min(payment, original_bid)
        
        slot_position = i + 1
        slot_positions[agent_name] = slot_position
        position_multiplier = 1.0 / slot_position
        user_type = np.random.choice(user_types)
        expected_revenue = (user_type['click_rate'] * position_multiplier * 
                          user_type['conversion_rate'] * user_type['value_per_conversion'])
        actual_revenue = np.random.normal(expected_revenue, expected_revenue * 0.2)
        revenues[agent_name] = max(0, actual_revenue)
    
    st.session_state.previous_bids = bids.copy()
    
    # Debug: Verify number of winners
    st.write(f"DEBUG: Number of slots: {num_slots}, Number of winners: {len(winners)}")
    
    return {
        'winners': [name for name, _ in winners],
        'payments': payments,
        'revenues': revenues,
        'slot_positions': slot_positions,
        'all_bids': dict(sorted_effective_bids)
    }

def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🏷️ Multi-Agent Ad Auction Platform</h1>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🔧 Auction Configuration")
        
        num_slots = st.number_input(
            "Number of Ad Slots", min_value=1, max_value=5, 
            value=st.session_state.auction_config['num_ad_slots']
        )
        st.session_state.auction_config['num_ad_slots'] = num_slots
        
        # Warn if num_ad_slots >= number of agents
        num_agents = len(st.session_state.agents)
        if num_slots >= num_agents and num_agents > 0:
            st.warning(f"⚠️ Number of slots ({num_slots}) is greater than or equal to the number of agents ({num_agents}). All agents may win a slot each round.")
        
        st.session_state.auction_config['max_rounds'] = st.number_input(
            "Maximum Rounds", min_value=1, max_value=20,
            value=st.session_state.auction_config['max_rounds']
        )
        st.session_state.auction_config['max_iterations_per_round'] = st.number_input(
            "Max Iterations per Round", min_value=1, max_value=10,
            value=st.session_state.auction_config['max_iterations_per_round']
        )
        st.session_state.auction_config['min_bid_increment'] = st.number_input(
            "Minimum Bid Increment", min_value=0.01, max_value=1.0, 
            value=st.session_state.auction_config['min_bid_increment'], step=0.01
        )
        st.session_state.auction_config['starting_budget'] = st.number_input(
            "Starting Budget per Agent", min_value=10.0, max_value=1000.0,
            value=st.session_state.auction_config['starting_budget']
        )
        
        st.divider()
        
        st.header("👥 Agent Management")
        
        with st.expander("Add New Agent"):
            new_agent_name = st.text_input("Agent Name")
            agent_type = st.selectbox("Agent Type", ["Human", "AI Bot", "Random Bot"])
            model_path = None
            quality_score = None
            if agent_type == "AI Bot":
                model_path = st.text_input("Path to Trained PPO Model (e.g., ppo_agent_agent1.pt)", 
                                         value="ppo_agent_agent1.pt")
            quality_score = st.number_input("Quality Score (0.5 to 1.0, leave blank for random)", 
                                          min_value=0.5, max_value=1.0, value=0.75, step=0.01, 
                                          key=f"qs_{new_agent_name}", format="%.2f")
            
            if st.button("Add Agent") and new_agent_name:
                if new_agent_name not in st.session_state.agents:
                    try:
                        st.session_state.agents[new_agent_name] = Agent(
                            new_agent_name, agent_type, 
                            st.session_state.auction_config['starting_budget'],
                            model_path if agent_type == "AI Bot" else None,
                            quality_score
                        )
                        st.success(f"Added agent: {new_agent_name} with Quality Score: {st.session_state.agents[new_agent_name].quality_score:.2f}")
                        st.rerun()
                    except FileNotFoundError:
                        st.error(f"Model file {model_path} not found!")
                    except RuntimeError as e:
                        st.error(f"Error loading model: {str(e)}")
                else:
                    st.error("Agent name already exists!")
        
        if st.session_state.agents:
            st.subheader("Current Agents")
            for agent_name, agent in st.session_state.agents.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{agent.name}** ({agent.agent_type})")
                        st.write(f"Budget: ${agent.budget:.2f}")
                        st.write(f"Quality Score: {agent.quality_score:.2f}")
                    with col2:
                        if st.button("❌", key=f"remove_{agent_name}"):
                            del st.session_state.agents[agent_name]
                            st.rerun()
        
        st.divider()
        
        st.header("🎮 Auction Controls")
        
        if not st.session_state.auction_active:
            if st.button("🚀 Start Auction", type="primary"):
                if len(st.session_state.agents) >= 2:
                    st.session_state.auction_active = True
                    st.session_state.current_round = 1
                    st.session_state.current_iteration = 1
                    st.session_state.current_bids = {}
                    st.session_state.previous_bids = {}
                    st.session_state.user_types = create_auction_environment()
                    st.rerun()
                else:
                    st.error("Need at least 2 agents to start auction!")
        else:
            if st.button("⏹️ Stop Auction", type="secondary"):
                st.session_state.auction_active = False
                st.rerun()
            
            if st.button("🔄 Reset Auction"):
                st.session_state.auction_active = False
                st.session_state.current_round = 1
                st.session_state.current_iteration = 1
                st.session_state.auction_history = []
                st.session_state.round_history = []
                st.session_state.current_bids = {}
                st.session_state.previous_bids = {}
                for agent in st.session_state.agents.values():
                    agent.budget = agent.initial_budget
                    agent.total_spent = 0.0
                    agent.total_revenue = 0.0
                    agent.slots_won = 0
                    agent.bid_history = []
                    agent.performance_history = []
                st.session_state.user_types = []
                st.rerun()
    
    if not st.session_state.agents:
        st.info("👈 Please add agents in the sidebar to begin the auction!")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Round", st.session_state.current_round)
    with col2:
        st.metric("Current Iteration", st.session_state.current_iteration)
    with col3:
        st.metric("Active Agents", len(st.session_state.agents))
    with col4:
        st.metric("Ad Slots Available", st.session_state.auction_config['num_ad_slots'])
    
    if st.session_state.auction_active:
        st.header(f"🔥 Round {st.session_state.current_round} - Iteration {st.session_state.current_iteration}")
        
        if 'current_user_types' not in st.session_state:
            st.session_state.current_user_types = create_auction_environment()
        
        with st.expander("📊 Current User Type Information"):
            user_df = pd.DataFrame(st.session_state.current_user_types)
            st.dataframe(user_df, use_container_width=True)
        
        st.subheader("💰 Place Your Bids")
        
        bid_cols = st.columns(len(st.session_state.agents))
        
        for i, (agent_name, agent) in enumerate(st.session_state.agents.items()):
            with bid_cols[i]:
                st.markdown(f'<div class="agent-card">', unsafe_allow_html=True)
                st.write(f"**{agent.name}**")
                st.write(f"Budget: ${agent.budget:.2f}")
                st.write(f"Type: {agent.agent_type}")
                st.write(f"Quality Score: {agent.quality_score:.2f}")
                
                if agent.agent_type == "Human":
                    max_bid = min(agent.budget, 50.0)
                    bid_amount = st.number_input(
                        f"Bid Amount", 
                        min_value=st.session_state.auction_config['min_bid_increment'],
                        max_value=max_bid,
                        value=st.session_state.auction_config['min_bid_increment'],
                        step=st.session_state.auction_config['min_bid_increment'],
                        key=f"bid_{agent_name}"
                    )
                    
                    if st.button(f"Submit Bid", key=f"submit_{agent_name}"):
                        if agent.place_bid(bid_amount):
                            st.session_state.current_bids[agent_name] = bid_amount
                            st.success(f"Bid submitted: ${bid_amount:.2f}")
                        else:
                            st.error("Insufficient budget!")
                
                else:
                    if agent_name not in st.session_state.current_bids:
                        if agent.agent_type == "Random Bot":
                            max_bid = min(agent.budget, 20.0)
                            bid_amount = np.random.uniform(
                                st.session_state.auction_config['min_bid_increment'], 
                                max_bid
                            )
                        else:
                            user_type_id = st.session_state.current_user_types[0]['id']
                            state = get_auction_state(agent, user_type_id)
                            bid_amount = agent.get_ppo_bid(state)
                        
                        if agent.place_bid(bid_amount):
                            st.session_state.current_bids[agent_name] = bid_amount
                            st.write(f"Auto-bid: ${bid_amount:.2f}")
                
                if agent_name in st.session_state.current_bids:
                    st.success(f"Current bid: ${st.session_state.current_bids[agent_name]:.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        all_agents_bid = len(st.session_state.current_bids) == len(st.session_state.agents)
        
        if all_agents_bid:
            st.divider()
            
            if st.button("🏁 Process Round Results", type="primary"):
                results = calculate_auction_results(
                    st.session_state.current_bids, 
                    st.session_state.current_user_types
                )
                
                for agent_name, agent in st.session_state.agents.items():
                    if agent_name in results['winners']:
                        payment = results['payments'][agent_name]
                        revenue = results['revenues'][agent_name]
                        agent.win_slot(payment, revenue)
                        agent.performance_history.append({
                            'round': st.session_state.current_round,
                            'won': True,
                            'payment': payment,
                            'revenue': revenue,
                            'profit': revenue - payment
                        })
                    else:
                        agent.performance_history.append({
                            'round': st.session_state.current_round,
                            'won': False,
                            'payment': 0,
                            'revenue': 0,
                            'profit': 0
                        })
                
                round_result = {
                    'round': st.session_state.current_round,
                    'iteration': st.session_state.current_iteration,
                    'bids': st.session_state.current_bids.copy(),
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.round_history.append(round_result)
                
                st.session_state.current_bids = {}
                st.session_state.current_round += 1
                st.session_state.current_user_types = create_auction_environment()
                
                if st.session_state.current_round > st.session_state.auction_config['max_rounds']:
                    st.session_state.auction_active = False
                    st.balloons()
                    st.success("🎉 Auction completed!")
                
                st.rerun()
    
    if st.session_state.current_bids:
        st.subheader("📋 Current Round Bids")
        bid_df = pd.DataFrame([
            {'Agent': name, 'Bid': f"${bid:.2f}", 'Budget Remaining': f"${st.session_state.agents[name].budget:.2f}", 
             'Quality Score': f"{st.session_state.agents[name].quality_score:.2f}"}
            for name, bid in st.session_state.current_bids.items()
        ])
        st.dataframe(bid_df, use_container_width=True)
    
    if st.session_state.round_history:
        st.header("📈 Auction Results & Analytics")
        
        latest_round = st.session_state.round_history[-1]
        
        st.subheader(f"🏆 Latest Round Results (Round {latest_round['round']})")
        
        if latest_round['results']['winners']:
            winner_data = []
            for winner in latest_round['results']['winners']:
                payment = latest_round['results']['payments'][winner]
                revenue = latest_round['results']['revenues'][winner]
                profit = revenue - payment
                quality_score = st.session_state.agents[winner].quality_score
                slot_position = latest_round['results']['slot_positions'][winner]
                winner_data.append({
                    'Slot': f"Slot {slot_position}",
                    'Winner': winner,
                    'Bid': f"${latest_round['bids'][winner]:.2f}",
                    'Quality Score': f"{quality_score:.2f}",
                    'Effective Bid': f"${latest_round['results']['all_bids'][winner]:.2f}",
                    'Payment': f"${payment:.2f}",
                    'Revenue': f"${revenue:.2f}",
                    'Profit': f"${profit:.2f}"
                })
                
            winner_df = pd.DataFrame(winner_data)
            st.dataframe(winner_df, use_container_width=True)
        else:
            st.info("No winners in the latest round.")
        
        st.subheader("📊 Agent Performance Summary (Cumulative Over All Rounds)")
        
        performance_data = []
        for agent_name, agent in st.session_state.agents.items():
            stats = agent.get_stats()
            performance_data.append(stats)
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            display_df = perf_df.copy()
            display_df['budget'] = display_df['budget'].apply(lambda x: f"${x:.2f}")
            display_df['total_spent'] = display_df['total_spent'].apply(lambda x: f"${x:.2f}")
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"${x:.2f}")
            display_df['roi'] = display_df['roi'].apply(lambda x: f"{x:.1f}%")
            display_df['avg_bid'] = display_df['avg_bid'].apply(lambda x: f"${x:.2f}")
            display_df['quality_score'] = display_df['quality_score'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roi = px.bar(
                    perf_df, x='name', y='roi',
                    title='Agent ROI Comparison (%)',
                    color='roi',
                    color_continuous_scale='RdYlGn'
                )
                fig_roi.update_layout(height=400)
                st.plotly_chart(fig_roi, use_container_width=True)
            
            with col2:
                fig_slots = px.pie(
                    perf_df, values='slots_won', names='name',
                    title='Slots Won Distribution (Cumulative)'
                )
                fig_slots.update_layout(height=400)
                st.plotly_chart(fig_slots, use_container_width=True)
        
        if len(st.session_state.round_history) > 1:
            st.subheader("📈 Round-by-Round Performance")
            
            timeline_data = []
            for round_data in st.session_state.round_history:
                round_num = round_data['round']
                for agent_name, agent in st.session_state.agents.items():
                    round_perf = next((p for p in agent.performance_history if p['round'] == round_num), None)
                    if round_perf:
                        timeline_data.append({
                            'Round': round_num,
                            'Agent': agent_name,
                            'Profit': round_perf['profit'],
                            'Revenue': round_perf['revenue'],
                            'Won': round_perf['won']
                        })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                fig_timeline = px.line(
                    timeline_df, x='Round', y='Profit', color='Agent',
                    title='Profit per Round by Agent'
                )
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                win_rate_data = timeline_df.groupby('Agent')['Won'].agg(['sum', 'count']).reset_index()
                win_rate_data['win_rate'] = (win_rate_data['sum'] / win_rate_data['count'] * 100).round(1)
                win_rate_data = win_rate_data.rename(columns={'sum': 'total_wins', 'count': 'total_rounds'})
                
                fig_winrate = px.bar(
                    win_rate_data, x='Agent', y='win_rate',
                    title='Win Rate by Agent (%)',
                    text='win_rate'
                )
                fig_winrate.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_winrate.update_layout(height=400)
                st.plotly_chart(fig_winrate, use_container_width=True)
    
    if st.session_state.round_history:
        st.divider()
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Performance Data"):
                export_data = {
                    'auction_config': st.session_state.auction_config,
                    'agents': {name: agent.get_stats() for name, agent in st.session_state.agents.items()},
                    'round_history': st.session_state.round_history,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📋 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"auction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Download Performance CSV"):
                performance_data = []
                for agent_name, agent in st.session_state.agents.items():
                    for perf in agent.performance_history:
                        perf_copy = perf.copy()
                        perf_copy['agent'] = agent_name
                        perf_copy['quality_score'] = agent.quality_score
                        performance_data.append(perf_copy)
                
                if performance_data:
                    csv_df = pd.DataFrame(performance_data)
                    csv_str = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_str,
                        file_name=f"agent_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()