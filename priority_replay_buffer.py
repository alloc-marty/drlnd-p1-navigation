import numpy as np
import random
from collections import namedtuple, deque
from typing import Callable

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class sliceable_deque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if (index.start is None or index.start > 0) else len(self) + index.start
            stop = index.stop if (index.stop is None or index.stop > 0) else len(self) + index.stop
            return itertools.islice(self, start, index.stop, index.step) if self else iter([])
        return deque.__getitem__(self, index)

class PriorityReplayBuffer:
    """Bounded-size buffer to store experience tuples and associated priority-weighting."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device, fn_calc_prios: Callable, initial_alpha=0.7):
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
        self.memory_exs = sliceable_deque(maxlen=buffer_size) #queue of experience tuples
        self.memory_prios = deque(maxlen=buffer_size)         #queue of priorities (redundant w/ prios_to_a, kept to avoid numerical error from power(1/alpha))
        self.memory_prios_to_a = deque(maxlen=buffer_size)    #queue of priorities raised to a
        self.N_added_since_calc = 0
        self.memory_size = buffer_size
        self.batch_size = batch_size
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = np.random.seed(seed)
        self._alpha = initial_alpha
        self.cached_p = None
        self.fn_calc_prios = fn_calc_prios

    def needs_calc_prios(self):
        return self.N_added_since_calc > 0

    def calc_prios(self):
        N_capacity_remain = self.memory_size - len(self.memory_prios)
        N_exs_dropped = max(self.N_added_since_calc - N_capacity_remain, 0)
        for _ in np.arange(min(N_exs_dropped, self.memory_size)):  #drop entries from prios queue for consistency w ex
            _ = self.memory_prios.popleft()
            _ = self.memory_prios_to_a.popleft()

#TODO: list() here premature?
        experiences = list(self.memory_exs[-self.N_added_since_calc:]) if self.N_added_since_calc < self.memory_size else self.memory_exs   #grab the experiences added since last calc
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).byte().to(self.device)

        for prio in self.fn_calc_prios((states, actions, rewards, next_states, dones)):
            self.memory_prios.append(prio)
            self.memory_prios_to_a.append(prio ** self._alpha)

        self.N_added_since_calc = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with the given priority (typically |td-error| + small bias 'e')"""
        e = self.Experience(state, action, reward, next_state, done)
        self.memory_exs.append(e)
        self.N_added_since_calc += 1
        self.cached_p = None

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

    
    def sample(self):
        """Randomly sample a batch of experiences from memory using priority-based distrib vs uniform per hyperparameter self.alpha [0..1]"""
        if not self.cached_p:
            self.cached_p = np.divide(self.memory_prios_to_a, sum(self.memory_prios_to_a))
        else:
            print('using cached_p')

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
