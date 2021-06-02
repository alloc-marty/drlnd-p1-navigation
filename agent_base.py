# -*- coding: utf-8 -*-
import random
import torch
import pickle
from collections import deque

class AgentBase():
    """ AgentBase: common stuff for all agents"""
    def init(self, state_size, action_size, seed, device:torch.device, is_training:bool):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        self.is_training = is_training

        self.N_steps = 0        # total number of steps trained
        self.N_episodes = 0     # total number of episodes trained
        self.ep_scores = deque()     # scores per episode

    def choose_action(self, state, params) -> int:
        "stateless function, selects an action given a state, using params"
        raise NotImplementedError()

    def _did_step(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def _load_training(self, unpickler:pickle.Unpickler):
        raise NotImplementedError()

    def _dump_training(self, pickler:pickle.Pickler):
        raise NotImplementedError()

    def did_step(self, state, action, reward, next_state, done):
        if not self.is_training:
            return
        self._did_step(state, action, reward, next_state, done)

    def load_training(self, infile):
        with pickle.Unpickler(infile) as unpickler:
            self._load_training(unpickler)
            self.N_steps = unpickler.load()
            self.N_episodes = unpickler.load()
            self.ep_scores = unpickler.load()


    def dump_training(self, outfile):
        with pickle.Pickler(outfile) as pickler:
            self._dump_training(pickler)
            pickler.dump(self.N_steps)
            pickler.dump(self.N_episodes)
            pickler.dump(self.ep_scores)

