import copy
import os
import random

random.seed(1)
from math import exp
import numpy as np

np.random.seed(1)
import torch

torch.manual_seed(1)
import time
from torch.distributions import Categorical
from scipy.stats import wasserstein_distance

device = 'cpu'
import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(1)


# Actor-PolicyNet
class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden):
        super(Actor, self).__init__()
        self.net = nn.ModuleList()
        for i in range(len(n_hidden)):
            if i == 0:
                self.net.append(nn.Linear(n_state, n_hidden[i]))
            else:
                self.net.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.net.append(nn.Tanh())
            # self.net.append(nn.ReLU())
        self.net.append(nn.Linear(n_hidden[-1], n_action))

    def forward(self, x):
        for m in self.net:
            x = m(x)
        x_ = (torch.ones_like(x) * 1e-3).to(x.device)
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), x_, x)
        out = F.softmax(x, dim=1)
        # print(out)

        return out


# Critic-ValueNet
class Critic(nn.Module):
    def __init__(self, n_state, n_out, n_hidden):
        super(Critic, self).__init__()
        # print(n_state)
        self.net = nn.ModuleList()
        for i in range(len(n_hidden)):
            if i == 0:
                self.net.append(nn.Linear(n_state, n_hidden[i]))
            else:
                self.net.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.net.append(nn.Tanh())
            # self.net.append(nn.ReLU())
        self.net.append(nn.Linear(n_hidden[-1], n_out))

    def forward(self, x):
        for m in self.net:
            x = m(x)
        # print(x) # len: batchsize
        return x


class Buffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []
        self.done = []
        self.cnt = 0
        self.batch_size = batch_size

    def store(self, state, action, reward, state_, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.state_.append(state_)
        self.done.append(done)
        self.cnt += 1

    def clean(self):
        self.cnt = 0
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []


# Agent Example
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

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_).to(device)
        target = self.critic(state_).detach().cpu().numpy() * (1 - done)
        target_list = []
        reward = np.array(self.buffer.reward)
        for r in reward[::-1]:
            target = r + target * self.gamma
            target_list.insert(0, target.tolist())
        target = torch.FloatTensor(target_list).to(device)
        return target

    def cal_advantage(self, target):
        state = torch.tensor(self.buffer.state, dtype=torch.float).to(device)
        v = self.critic(state)
        adv = (target - v).detach()
        return adv

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state).to(device)
        v = self.critic(state)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(v, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    # adaptive trust region clipping
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
        threshold = self.threshold_base * exp(-self.attenuation_factor * self.learn_step)
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

    # store the model
    def save(self, prefix=None, weight=None):
        if weight is None:
            dir_name = time.strftime('%m-%d-%H-%M')
        else:
            str_w = [str(round(100 * w)) for w in weight.reshape(-1, ).tolist()]
            dir_name = 'w' + '_'.join(str_w)

        path = '/'.join([self.save_path, dir_name])
        if not os.path.exists(path):
            os.makedirs(path)
        if prefix is None:
            actor_file = 'actor.pkl'
            critic_file = 'critic.pkl'
        else:
            actor_file = prefix + 'actor.pkl'
            critic_file = prefix + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.critic.net.state_dict(), critic_file)

    # load the stored model
    def load(self, pkl_list):
        self.actor.net.load_state_dict(torch.load(pkl_list[0], map_location=torch.device(device)))
        self.critic.net.load_state_dict(torch.load(pkl_list[1], map_location=torch.device(device)))
