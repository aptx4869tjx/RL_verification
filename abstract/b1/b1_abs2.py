import os
import sys

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# 获取文件所在的当前路径
from abstract_env.b1 import B1Env
from verify.divide_tool import initiate_divide_tool, str_to_list, initiate_divide_tool_rtree

script_path = os.path.split(os.path.realpath(__file__))[0]
pt_file0 = os.path.join(script_path, "b1_abs-actor2.pt")
pt_file1 = os.path.join(script_path, "b1_abs-critic2.pt")
pt_file2 = os.path.join(script_path, "b1_abs-actor-target2.pt")
pt_file3 = os.path.join(script_path, "b1_abs-critic-target2.pt")

hiden_size = 100

state_space = [[-2.5, -2.5], [2.5, 2.5]]
initial_intervals = [0.01, 0.01]
env = B1Env()
env.reset()


# env.render()

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 1)
        self.linear1.bias.data.zero_()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 1)
        self.linear2.bias.data.zero_()
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 1)
        self.linear3.bias.data.zero_()

    def forward(self, s):
        # x = F.relu(self.linear1(s))
        # x = F.relu(self.linear2(x))
        # x = torch.tanh(self.linear3(x))
        x = torch.tanh(self.linear1(s))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        # x = torch.sigmoid(self.linear1(s))
        # x = torch.sigmoid(self.linear2(x))
        # x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


params = {
    'env': env,
    'gamma': 0.99,
    'actor_lr': 0.0001,
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 20000,
    'batch_size': 32,
}


class Agent(object):
    def __init__(self, divide_tool):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5

        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]

        self.divide_tool = divide_tool
        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.noisy = [0.0005, 0.0005]

    def update_egreed(self):  # 更新贪心参数e,该函数有问题？？？？
        self.e_greed = max(0.01, self.e_greed - 0.001)

    def reset(self):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5
        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]

        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.noisy = [0.0005, 0.0005]

    def save(self):  # 保存网络的参数数据
        torch.save(self.actor.state_dict(), pt_file0)
        torch.save(self.critic.state_dict(), pt_file1)
        torch.save(self.actor_target.state_dict(), pt_file2)
        torch.save(self.critic_target.state_dict(), pt_file3)
        # print(pt_file + " saved.")

    def load(self):  # 加载网络的参数数据
        self.actor.load_state_dict(torch.load(pt_file0))
        self.network.load_state_dict(torch.load(pt_file0))
        self.critic.load_state_dict(torch.load(pt_file1))
        self.actor_target.load_state_dict(torch.load(pt_file2))
        self.critic_target.load_state_dict(torch.load(pt_file3))
        print(pt_file3 + " loaded.")

    def act(self, s0):
        abs = str_to_list(self.divide_tool.get_abstract_state(s0))
        # s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        s0 = torch.tensor(abs, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def add_noisy(self, obs):

        obs[0] += random.gauss(0, self.noisy[0])
        obs[1] += random.gauss(0, self.noisy[1])

        return obs

    def update_noisy(self):
        # noisy = [0.005, 0.01, 0.0005, 0.01]
        step_length = [0.0005, 0.0005]
        for i in range(len(step_length)):
            self.noisy[i] += step_length[i]

    def evaluate_with_noisy(self, episode):
        iteration_mean_reward_list = []
        for j in range(300):
            reward_list = []
            print('evaluate iteration: ', j)
            for i in range(episode):
                total_reward = 0
                obs = env.reset()
                obs = self.add_noisy(obs)
                obs = clip(obs)
                step_size = 0
                for t in range(500):
                    act = self.act(obs)
                    obs, reward, done, _ = env.step(act)
                    obs = self.add_noisy(obs)
                    obs = clip(obs)
                    total_reward += reward
                    step_size += 1
                    if done:
                        break
                reward_list.append(total_reward)
                # print('EVALUATE: ', i, step_size)
            print('AVG REWARD: ', np.mean(reward_list))
            iteration_mean_reward_list.append(np.array(reward_list))
            self.update_noisy()
        return np.array(iteration_mean_reward_list)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


def clip(state):
    state_space1 = [[-2.49999, -2.49999], [2.49999, 2.49999]]
    state[0] = np.clip(state[0], state_space1[0][0], state_space1[1][0])
    state[1] = np.clip(state[1], state_space1[0][1], state_space1[1][1])
    return state


def evaluate(agent):
    min_reward = 0
    crash = False
    reward_list = []
    for l in range(1000):
        reward = 0
        s0 = env.reset()
        reach = False

        for step in range(10000):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            if done:
                print(l, 'reach goal', s1, step, end='----')
                reach = True
                break
            reward += r1
            s0 = s1
        print(reward)
        reward_list.append(reward)
        if reward <= -600:
            crash = True
        if not reach:
            print('Not reach goal!!!--------------------------------')
    print('crash: ', crash)
    print('avg reward: ', np.mean(reward_list))
    return np.array(reward_list)


def train_model(agent):
    reward_list = []
    for j in range(20):
        agent.reset()
        for episode in range(300):
            agent.update_egreed()
            s0 = env.reset()
            episode_reward = 0
            ab_s = agent.divide_tool.get_abstract_state(s0)
            step_size = 0
            for step in range(150):
                # env.render()
                # if np.random.rand() < agent.e_greed:
                #     a0 = [(np.random.rand() - 0.5) * 2]
                # else:
                a0 = agent.act(s0)
                s1, r1, done, _ = env.step(a0)
                step_size += 1
                next_abs = agent.divide_tool.get_abstract_state(s1)

                agent.put(str_to_list(ab_s), a0, r1, str_to_list(next_abs))

                episode_reward += r1
                s0 = s1
                ab_s = next_abs
                if step % 4 == 0:
                    agent.learn()
                if done:
                    break
            if episode % 5 == 4:
                agent.save()
            reward_list.append(episode_reward)
            print(episode, ': ', episode_reward, step_size)
            if episode >= 100 and np.min(reward_list[-4:]) >= -125:
                #     min_reward = evaluate(agent)
                #     if min_reward > -30:
                agent.save()
                return [], []

            # divide_tool = initiate_divide_tool(state_space, initial_intervals)


if __name__ == "__main__":
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], 'b1+abstraction')
    agent = Agent(divide_tool)
    # agent.load()
    train_model(agent)
    agent.load()
    res_list = evaluate(agent)
    print(len(res_list), 'avg: ', np.mean(res_list))
