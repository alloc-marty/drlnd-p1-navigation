import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PriorityReplayBuffer:
    """Bounded-size buffer to store experience tuples and associated priority-weighting."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, initial_alpha=0.7):
        """Initialize a PriorityReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory_exs = deque(maxlen=buffer_size+1)           #queue of experience tuples
        self.memory_prios = deque(maxlen=buffer_size+1)         #queue of priorities (redundant w/ prios_to_a, kept to avoid numerical error from power(1/alpha))
        self.memory_prios_to_a = deque(maxlen=buffer_size+1)    #queue of priorities raised to a
        self.sum_prios_to_a = 0                                 #sum of memory_prios_to_a 
        self.memory_size = buffer_size
        self.batch_size = batch_size
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = np.random.seed(seed)
        self._alpha = initial_alpha
        self.cached_p = None

    
    def add(self, state, action, reward, next_state, done, priority = 1.0):
        """Add a new experience to memory with the given priority (typically |td-error| + small bias 'e')"""
        e = self.Experience(state, action, reward, next_state, done)
        self.memory_exs.append(e)
        self.memory_prios.append(priority)
        prio_to_a = priority ** self._alpha
        self.memory_prios_to_a.append(prio_to_a)
        self.sum_prios_to_a += prio_to_a        
        self.cached_p = None
        if len(self.memory_exs) > self.memory_size: #at capacity, pop an entry
            _ = self.memory_exs.pop()
            _ = self.memory_prios.pop()
            self.sum_prios_to_a -= self.memory_prios_to_a.pop()


    def get_alpha(self):
        return self._alpha
    
    def set_alpha(self, alpha):
        """Update alpha
        
        Params
        ======
            alpha (float): new alpha value [0.0 (equiprobable random) .. 1.0 (random, fully based on priorities)]
        """
        if self._alpha == alpha:
            return
        self.cached_p = None
        self._alpha = alpha
        self.memory_prios_to_a = deque(np.power(self.memory_prios, self._alpha)) #recalc
        self.sum_prios_to_a = np.sum(self.memory_prios_to_a)

    
    def sample(self):
        """Randomly sample a batch of experiences from memory using priority-based distrib vs uniform per hyperparameter self.alpha [0..1]"""
        if not self.cached_p:
            self.cached_p = np.divide(self.memory_prios_to_a, self.sum_prios_to_a)
        experiences_i = np.random.choice(len(self.memory_exs), replace=False, size=self.batch_size, p=self.cached_p)
        experiences = [self.memory_exs[i] for i in experiences_i]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).byte().to(self.device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory_exs)
