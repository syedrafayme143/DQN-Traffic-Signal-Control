import numpy as np 
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_dim, fixed_action_space, TL_list, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY):
        self.state_size = state_size + len(TL_list) if fixed_action_space else state_size
        self.action_size = action_size
        self.fixed_action_space = fixed_action_space
        self.TL_list = TL_list
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR = LR
        self.UPDATE_EVERY = UPDATE_EVERY

        print(hidden_dim)
        hidden_dim.insert(0, self.state_size)
        hidden_dim.append(self.action_size)

        self.qnetwork_local = QNetwork().to(device)
        print(self.qnetwork_local)
        self.qnetwork_target = QNetwork().to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = SequentialMemory(self.BUFFER_SIZE, self.BATCH_SIZE)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        reward = sum(list(reward.values()))
        if self.fixed_action_space:
            for tl_index, tl_id in enumerate(self.TL_list):
                one_hot = np.zeros(len(self.TL_list))
                np.put(one_hot, tl_index, 1)
                state_one_hot = np.array(one_hot.tolist() + state)
                next_state_one_hot = np.array(one_hot.tolist() + next_state)
                self.memory.add(state_one_hot, action[tl_id], reward, next_state_one_hot, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        if self.fixed_action_space:
            action_dict = dict.fromkeys(self.TL_list, 0)
            action_values_dict = dict.fromkeys(self.TL_list, 0)

            for tl_index, tl_id in enumerate(self.TL_list):
                one_hot = np.zeros(len(self.TL_list))
                np.put(one_hot, tl_index, 1)
                state_one_hot = np.array(one_hot.tolist() + state.tolist())
                state_one_hot = torch.from_numpy(state_one_hot).float().unsqueeze(0).to(device)

                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local(state_one_hot)
                self.qnetwork_local.train()

                action_values_dict[tl_id] = action_values.tolist()

                # TASK 1: Exploration Strategy
                if random.random() > eps:
                    action_dict[tl_id] = np.argmax(action_values.cpu().data.numpy())
                else:
                    action_dict[tl_id] = random.choice(np.arange(self.action_size))
            return action_values_dict, action_dict
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # TASK 1: Exploration vs. Exploitation
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # TASK 2: Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # TASK 2: Compute Q targets using Bellman equation
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # TASK 3: Compute MSE Loss
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_models(self, path):
        print('... saving models ...')
        self.qnetwork_local.save_checkpoint(path)

    def load_models(self, path):
        print('... loading models ...')
        self.qnetwork_local.load_checkpoint(path)


class RingBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
            self.data.append(v)
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
            self.data[(self.start + self.length - 1) % self.maxlen] = v
        else:
            raise RuntimeError()


class SequentialMemory:
    def __init__(self, buffer_size, batch_size):
        self.memory = RingBuffer(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory.data, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory.data)

