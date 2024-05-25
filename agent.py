import copy
import os
import random
from math import exp
import numpy as np
import torch
import time
from torch.distributions import Categorical
from scipy.stats import wasserstein_distance
device = 'cpu'
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import Actor, Critic, Buffer

# An agent example containing key process
class PPO_Agent:
    def __init__(self, args):
        n_hidden = [32, 64, 32]
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.save_path = args.sa_ckpt_path
        self.state_dim = args.sa_state_dim + args.objective
        self.action_dim = args.sa_action_space
        self.actor = Actor(self.state_dim, self.action_dim, n_hidden).to(device)
        self.old_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(self.state_dim, 1, n_hidden).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.w = None
        self.learn_step = 0  # record iteration number
        self.critic_loss = []
        self.actor_loss = []

        # parameters for adaptive trust region clipping
        self.threshold_base = args.threshold_base
        self.attenuation_factor = args.attenuation_factor
        self.rollback_factor = args.rollback_factor

        # parameter for comparing clipping method
        self.clip_method = args.clip

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        prob = self.actor(state).squeeze(0)
        dist = Categorical(prob)
        action = dist.sample().item() 
        return action

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state).to(device)
        v = self.critic(state)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(v, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    # proposed adaptive trust region clipping
    def wasserstein_clip(self, old_prob, prob, ratio):
        threshold = self.update_threshold()
        # print(threshold)
        old_prob_lst = old_prob.detach().numpy()
        prob_lst = prob.detach().numpy()

        clip_res_lst = []
        for i in range(len(prob_lst)):
            wasserstein_d = wasserstein_distance(old_prob_lst[i], prob_lst[i])
            if wasserstein_d >= threshold:
                clip_res_lst.append([-self.rollback_factor * ratio[i][0].detach().item()])
            else:
                clip_res_lst.append([ratio[i][0].detach().item()])
        return torch.FloatTensor(clip_res_lst).to(device)

    # threshold reduce
    def update_threshold(self):
        threshold = self.threshold_base * exp(-self.attenuation_factor * (self.learn_step + 1))
        # print(f"ra_threshold{threshold}")
        return threshold

    def actor_update(self, target):
        adv = self.cal_advantage(target).reshape(-1, 1)
        state = torch.FloatTensor(self.buffer.state).to(device)
        action = torch.LongTensor(self.buffer.action).view(-1, 1).to(device)
        # print(len(state))

        prob = self.actor(state)
        old_prob = self.old_actor(state)
        prob1 = prob.gather(1, action)
        old_prob1 = old_prob.gather(1, action)

        ratio = torch.exp(torch.log(prob1) - torch.log(old_prob1))
        surr = ratio * adv

        # compare the two different clipping function
        if self.clip_method == "PPO":
            # PPO-Clip
            loss = - torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv))
        else:
            # Adaptive trust region-Clip
            loss = - torch.mean(torch.min(surr, self.wasserstein_clip(old_prob, prob, ratio) * adv))
        self.actor_optim.zero_grad()
        loss.backward()
        return loss.item()

    # update the actor/critic network
    def learn(self, state_, done):
        if state_ is None:
            raise Exception('state_ is None')
        target = self.cal_target(state_, done)
        self.old_actor.load_state_dict(self.actor.state_dict())
        actor_loss = 0
        critic_loss = 0
        for _ in range(self.a_update_step):
            actor_loss += self.actor_update(target)
        for _ in range(self.c_update_step):
            critic_loss += self.critic_update(target)
        self.buffer.clean()  # clean the buffer after learning
        self.learn_step += 1  # update the learning step(iteration number)
        actor_loss = actor_loss / self.a_update_step
        critic_loss = critic_loss / self.c_update_step
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        return actor_loss, critic_loss

    # config the weight
    def cweight(self, weight):
        self.w = weight
