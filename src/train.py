from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from collections import deque
import numpy as np
import random
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import pickle
import os
os.environ['KMP_WARNINGS'] = 'off'
import warnings
warnings.filterwarnings('ignore')
from numba import njit
from numba import int32, float32
from numba.experimental import jitclass
import logging


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

#Segment Tree

spec_SumSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
@jitclass(spec=spec_SumSegmentTree)
class SumSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _sum_helper(self.tree, start, end, node, node_start, node_end)

    def sum(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.sum_helper(start, end, 1, 0, self.capacity - 1)

    def retrieve(self, upperbound: float) -> int:
        return _sum_retrieve_helper(self.tree, 1, self.capacity, upperbound)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _sum_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


spec_MinSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
INF = float('inf')
@jitclass(spec=spec_MinSegmentTree)
class MinSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.array([INF for _ in range(2 * capacity)], dtype=np.float32)

    def min_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _min_helper(self.tree, start, end, node, node_start, node_end)

    def min(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.min_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _min_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


@njit(cache=True)
def _sum_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _sum_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _sum_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _sum_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _sum_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return a + b


@njit(cache=True)
def _min_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _min_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _min_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _min_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _min_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            if a < b:
                return a
            else:
                return b


@njit(cache=True)
def _sum_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx = idx // 2


@njit(cache=True)
def _min_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        a = tree[2 * idx]
        b = tree[2 * idx + 1]
        if a < b:
            tree[idx] = a
        else:
            tree[idx] = b
        idx = idx // 2


@njit(cache=True)
def _sum_retrieve_helper(tree: np.ndarray, idx: int, capacity: int, upperbound: float) -> int:
    while idx < capacity: # while non-leaf
        left = 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx = 2 * idx
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PER = True
PREV_EPISODES = 0
EPISODES = 1000
MODEL_NAME = "ddqn_512"
if PER:
    SHORT_PATH = "/content/gdrive/MyDrive/rlassign/" + "bisPER_" + MODEL_NAME
else:
    SHORT_PATH = "/content/gdrive/MyDrive/rlassign/" + "" + MODEL_NAME
LOAD_PATH = SHORT_PATH + f"_{EPISODES}.pth"
reward_scaler = 1e8



class ReplayBuffer_:
    '''A simple numpy replay buffer.'''
    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
    ):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size), dtype=np.float32)
        self.rews_buf = np.zeros((size), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr = 0
        self.size = 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer_):
    '''Prioritized Replay buffer.'''
    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.000005,
    ):
        '''Initialization.'''
        assert alpha >= 0

        super().__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        '''Store experience and priority.'''
        super().store(obs, act, rew, next_obs, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        '''Sample a batch of experiences.'''
        assert len(self) >= self.batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = self._calculate_weights(indices, self.beta)

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        '''Update priorities of sampled transitions.'''
        assert len(indices) == len(priorities)
        _update_priorities_helper(indices, priorities, self.sum_tree, self.min_tree, self.alpha)
        self.max_priority = max(self.max_priority, priorities.max())

    def _sample_proportional(self) -> np.ndarray:
        '''Sample indices based on proportions.'''
        return _sample_proportional_helper(self.sum_tree, len(self), self.batch_size)

    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        '''Calculate the weights of the experiences'''
        return _calculate_weights_helper(indices, beta, self.sum_tree, self.min_tree, len(self))


@njit(cache=True)
def _sample_proportional_helper(
    sum_tree: SumSegmentTree,
    size: int,
    batch_size: int,
) -> np.ndarray:
    indices = np.zeros(batch_size, dtype=np.int32)
    p_total = sum_tree.sum(0, size - 1)
    segment = p_total / batch_size

    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)
        upperbound = np.random.uniform(a, b)
        idx = sum_tree.retrieve(upperbound)
        indices[i] = idx

    return indices


@njit(cache=True)
def _calculate_weights_helper(
    indices: np.ndarray,
    beta: float,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    size: int,
) -> np.ndarray:

    weights = np.zeros(len(indices), dtype=np.float32)

    for i in range(len(indices)):

        idx = indices[i]

        # get max weight
        p_min = min_tree.min() / sum_tree.sum()
        max_weight = (p_min * size) ** (-beta)

        # calculate weights
        p_sample = sum_tree[idx] / sum_tree.sum()
        weight = (p_sample * size) ** (-beta)
        weight = weight / max_weight

        weights[i] = weight

    return weights


@njit(cache=True)
def _update_priorities_helper(
    indices: np.ndarray,
    priorities: np.ndarray,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    alpha: float,
) -> None:

    for i in range(len(indices)):
        idx = indices[i]
        priority = priorities[i]
        sum_tree[idx] = priority ** alpha
        min_tree[idx] = priority ** alpha

#Memory


def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> dict:
    if memory is None:
        return {}
    per_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
        'max_priority', 'tree_ptr', 'alpha',
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result['sum_tree'] = dict(
        capacity = memory.sum_tree.capacity,
        tree = memory.sum_tree.tree,
    )
    result['min_tree'] = dict(
        capacity = memory.min_tree.capacity,
        tree = memory.min_tree.tree,
    )
    return result

# Réseau de neurones, double dqn
class QNetwork(nn.Module):


    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf),
            nn.LeakyReLU(),
            nn.LayerNorm(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.LayerNorm(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.LayerNorm(nf),
            nn.Linear(nf, out_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.log(x)
        return self.layers(x)
    
# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)
    


# Reinforcement Learning Project Agent
class ProjectAgent:
    def __init__(self, gamma=0.99, lr=1e-3, buffer_size=100000, batch_size=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.996,
                 grad_clip=1000.0, double_dqn=True, hidden_dim=512,
                 # PER parameters
                per: bool = PER,
                alpha: float = 0.2,
                beta: float = 0.6,
                beta_increment_per_sampling: float = 5e-6,
                prior_eps: float = 1e-6,
):
        # Device: cpu / gpu
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        logging.info(f'device: {self.device}')
        self.double_dqn = double_dqn

        self.state_size = env.unwrapped.observation_space.shape[0]
        self.action_size = env.unwrapped.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.hidden_dim = hidden_dim
        self.q_network = QNetwork(self.state_size, self.hidden_dim, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.hidden_dim, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.criterion = "smooth_l1_loss"

        #self.memory = ReplayBuffer(buffer_size, batch_size)
        # PER memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.alpha = alpha
            self.beta = beta
            self.beta_increment_per_sampling = beta_increment_per_sampling
            self.memory = PrioritizedReplayBuffer(self.state_size, buffer_size, batch_size, alpha, beta, beta_increment_per_sampling)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size)

        self.rewards = []
        self.losses = []

    def act(self, observation: np.ndarray, use_random: bool=False) -> int:
        if (use_random) & (random.random() < self.epsilon):
            return random.randint(0, self.action_size - 1)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            selected_action = self.q_network(observation).argmax()

        return selected_action.detach().cpu().numpy()
    
    def save(self, path: str) -> None:
        # Enregistrement des poids du modèle
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "memory": _memory
                    }, path)
            logging.info(f"Agent's state saved to {path}")
            # enregistrment dans un fichier Json
            config_path = path[:-4] + "_config.pkl"
            agent_config = {
                "device": str(self.device),
                "double_dqn": self.double_dqn,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_size" : self.buffer_size,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "hidden_dim": self.hidden_dim,
                "grad_clip": self.grad_clip,
                "criterion": self.criterion,
                "rewards": self.rewards,
                "losses": self.losses,
                "per": self.per,
                "alpha": self.alpha,
                "beta": self.beta,
                "beta_increment_per_sampling": self.beta_increment_per_sampling,
                "prior_eps": self.prior_eps
            }
            with open(config_path, "wb") as f:
                pickle.dump(agent_config, f)
            logging.info(f"Agent's config saved to {config_path}")
        else:
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                    }, path)
            logging.info(f"Agent's state saved to {path}")
            # enregistrement dans un fichier json
            config_path = path[:-4] + "_config.pkl"
            agent_config = {
                "device": str(self.device),
                "double_dqn": self.double_dqn,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_size" : self.buffer_size,
                "buffer" : list(self.memory.buffer),
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "hidden_dim": self.hidden_dim,
                "grad_clip": self.grad_clip,
                "criterion": self.criterion,
                "rewards": self.rewards,
                "losses": self.losses,
            }

            with open(config_path, "wb") as f:
                pickle.dump(agent_config, f)
            logging.info(f"Agent's config saved to {config_path}")

    def load(self) -> None:
        # Chargement des poids du modèle et de l'état de l'optimiseur
        checkpoint_data = torch.load(model_path = "dqn_model.pth", weights_only=False)
        config_file_path = "dqn_model_config.pkl"

        with open(config_file_path, "rb") as file:
            saved_config = pickle.load(file)

        # Recharger les réseaux de neurones
        self.q_network.load_state_dict(checkpoint_data["q_network"])
        self.target_network.load_state_dict(checkpoint_data["target_network"])
        self.target_network.eval()

        # Restaurer l'état de l'optimiseur
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])

        # Configuration de l'appareil de calcul
        self.device = torch.device("cpu")

        # Chargement des hyperparamètres depuis la sauvegarde
        for param in ["double_dqn", "state_size", "action_size", "gamma", "lr",
                        "batch_size", "buffer_size", "epsilon", "epsilon_min",
                        "epsilon_decay", "hidden_dim", "grad_clip", "criterion"]:
            setattr(self, param, saved_config[param])

        self.rewards = saved_config["rewards"]
        self.losses = saved_config["losses"]

        # Gestion de la mémoire d'expérience
        if self.per:
            for attr in ["alpha", "beta", "beta_increment_per_sampling", "prior_eps"]:
                setattr(self, attr, saved_config[attr])

            for key, value in checkpoint_data['memory'].items():
                if key in ['sum_tree', 'min_tree']:
                    tree_obj = getattr(self.memory, key)
                    tree_obj.capacity = value['capacity']
                    tree_obj.tree = value['tree']
                else:
                    setattr(self.memory, key, value)
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
            self.memory.buffer = deque(saved_config["buffer"], maxlen=self.buffer_size)

        logging.info(f"Agent state restored from {"dqn_model.pth"}")
        logging.info(f"Configuration restored from {config_file_path}")

    def load_specified(self, checkpoint_path):
        # Chargement spécifique depuis un chemin donné
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        config_file_path = checkpoint_path.replace('.pth', '_config.pkl')

        with open(config_file_path, "rb") as file:
            saved_config = pickle.load(file)

        # Restauration des réseaux et de l'optimiseur
        self.q_network.load_state_dict(checkpoint_data["q_network"])
        self.target_network.load_state_dict(checkpoint_data["target_network"])
        self.target_network.eval()
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])

        # Configuration de l'appareil de calcul
        self.device = torch.device("cpu")

        # Recharger les hyperparamètres
        for param in ["double_dqn", "state_size", "action_size", "gamma", "lr",
                        "batch_size", "buffer_size", "epsilon", "epsilon_min",
                        "epsilon_decay", "hidden_dim", "grad_clip", "criterion"]:
            setattr(self, param, saved_config[param])

        self.rewards = saved_config["rewards"]
        self.losses = saved_config["losses"]

        # Gestion de la mémoire d'expérience
        if self.per:
            for attr in ["alpha", "beta", "beta_increment_per_sampling", "prior_eps"]:
                setattr(self, attr, saved_config[attr])

            for key, value in checkpoint_data['memory'].items():
                if key in ['sum_tree', 'min_tree']:
                    tree_obj = getattr(self.memory, key)
                    tree_obj.capacity = value['capacity']
                    tree_obj.tree = value['tree']
                else:
                    setattr(self.memory, key, value)
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
            self.memory.buffer = deque(saved_config["buffer"], maxlen=self.buffer_size)

        logging.info(f"Agent state restored from {checkpoint_path}")
        logging.info(f"Configuration restored from {config_file_path}")

    def append_rewards(self, reward):
        self.rewards.append(reward)

    def append_losses(self, loss):
        self.losses.append(loss)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            selected_action = self.q_network(state).argmax()

        return selected_action.detach().cpu().numpy()

    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        device = self.device  # for shortening the following lines
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(dones.reshape(-1, 1)).to(device)

        curr_q_values = self.q_network(states).gather(1, actions)
        if not self.double_dqn:
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_values = self.target_network(next_states).gather(
                1, self.q_network(next_states).argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - dones
        targets = (rewards + self.gamma * next_q_values * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_values, targets, reduction='none')
        loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def learn_per(self):
        '''Update the model for PER dqn'''
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch()
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Return categorical dqn loss.'''
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.q_network(state).gather(1, action)
        if not self.double_dqn:
            next_q_value = self.target_network(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = self.target_network(next_state).gather(
                1, self.q_network(next_state).argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction='none')

        return elementwise_loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


#charger le modèle


