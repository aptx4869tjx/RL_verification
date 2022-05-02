#!/usr/bin/python
# coding=utf-8
import bisect
import copy
import os
import random
import sys
from collections import deque, namedtuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rtree import index
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from verify.divide_tool import DivideTool, str_to_list, initiate_divide_tool_rtree


rindex = '0'

# 获取文件所在的当前路径
script_path = os.path.split(os.path.realpath(__file__))[0]
# 判断是否使用double DQN
USE_DBQN = False
# 生成需要保存的文件名称以及路径
pt_file = os.path.join(script_path, "dqn" + rindex + ".pt")
# 创建测试环境testbed
env = gym.make("CartPole-v1")
#
# env = MyCartPoleEnv()
# env = CartPoleEnvInf(None)
# 去掉step的限制，可以无限跑
env = env.unwrapped
# env.tau=0.001
# 创建需要保存的数据结构
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)
OBS_N = env.observation_space.shape[0]  # 整型，状态空间维数
ACT_N = env.action_space.n  # 整型，动作空间个数
HIDDEN_N = 64  # 隐藏层节点数目
MEMORY_CAPACITY = 10000  # 经验回放盒大小
WARMUP_SIZE = 5120  # 经验回放盒大于该数目，才会开始采样学习，否则不采样
BATCH_SIZE = 256  # 每次采样数目
MODEL_SYNC_COUNT = 8  # 目标网络的更新频率
LEARNING_RATE = 1e-3  # 学习率
LEARN_FREQ = 8  # 并不是每次都学习，每8次才会学习一次
WEIGHT_DECAY = 0
GAMMA = 0.99  # 累计衰减系数
E_GREED = 0.1  # 贪心算法的权重e
E_GREED_DEC = 1e-4  # e每次下降的次数
E_GREED_MIN = 0.01  # e的最小值
EPISODES_NUM = 5000  # 最大回合数目
LEAST_EPISODES_NUM = 1000  # 每次训练，最小的回合数目
TRAIN_MAX_STEP = 500  # 训练阶段的最大步数
EVALUTE_MAX_STEP = 1000  # 评价阶段的最大步数。

state_space = [[-4.8, -10, -0.42, -10], [4.8, 10, 0.42, 10]]
initial_intervals = [0.01, 0.02, 0.001, 0.02]


# initial_intervals = [0.005, 0.01, 0.001, 0.01]
# initial_intervals = [0.01, 0.1, 0.005, 0.1]


class Model(nn.Module):  # 神经网络的类
    def __init__(self):  # 网络结构及其初始化
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.zero_()

    def forward(self, obs):  # 隐藏层的激活函数
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        action_val = self.fc4(obs)
        return action_val
        # 返回的是最终的每个action的数值


class Agent:
    def __init__(self, divide_tool):
        self.model = Model()  # 当前网络
        self.network = Model()
        self.target_model = Model()  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.memory = deque(maxlen=MEMORY_CAPACITY)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数
        self.e_greed = E_GREED
        self.update_count = 0  # 更新次数
        self.noisy = [0, 0, 0, 0]
        self.divide_tool = divide_tool

    def reset(self):
        self.model = Model()  # 当前网络
        self.network = Model()
        self.target_model = Model()  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.memory = deque(maxlen=MEMORY_CAPACITY)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数
        self.e_greed = E_GREED
        self.update_count = 0
        self.noisy = [0, 0, 0, 0]

    def update_egreed(self):  # 更新贪心参数e,该函数有问题？？？？
        self.e_greed = max(E_GREED_MIN, self.e_greed - E_GREED_DEC)

    def predict(self, obs):  # 选取最大的action
        abs = str_to_list(self.divide_tool.get_abstract_state(obs))
        q_val = self.model(torch.FloatTensor(abs)).detach().numpy()
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return choice_list[0]

    def sample(self, obs):  # 决定是随机选，还是选最大
        if np.random.rand() < self.e_greed:
            return np.random.choice(ACT_N)
        return self.predict(obs)

    def store_transition(self, trans):  # 放入经验回放盒
        self.memory.append(trans)

    def learn(self):  # 该函数实现了：采样，计算损失，反向传播，更新参数
        assert WARMUP_SIZE >= BATCH_SIZE
        if len(self.memory) < WARMUP_SIZE:
            return
        # 从经验回放盒中进行采样
        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*(zip(*batch)))
        s0 = torch.FloatTensor(batch.state)
        a0 = torch.LongTensor(batch.action).unsqueeze(1)
        r1 = torch.FloatTensor(batch.reward)
        s1 = torch.FloatTensor(batch.next_state)
        d1 = torch.LongTensor(batch.done)

        q_pred = self.model(s0).gather(1, a0).squeeze()
        with torch.no_grad():
            if USE_DBQN:
                acts = self.model(s1).max(1)[1].unsqueeze(1)
                q_target = self.target_model(s1).gather(1, acts).squeeze(1)
            else:
                q_target = self.target_model(s1).max(1)[0]

            q_target = r1 + GAMMA * (1 - d1) * q_target
        loss = self.loss_func(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % MODEL_SYNC_COUNT == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self):  # 保存网络的参数数据
        torch.save(self.model.state_dict(), pt_file)
        # print(pt_file + " saved.")

    def load(self):  # 加载网络的参数数据
        self.model.load_state_dict(torch.load(pt_file))
        self.network.load_state_dict(torch.load(pt_file))
        self.target_model.load_state_dict(self.model.state_dict())
        print(pt_file + " loaded.")

    def add_noisy(self, obs):

        obs[0] += random.gauss(0, self.noisy[0])
        obs[1] += random.gauss(0, self.noisy[1])
        obs[2] += random.gauss(0, self.noisy[2])
        obs[3] += random.gauss(0, self.noisy[3])
        return obs

    def update_noisy(self):
        # noisy = [0.005, 0.01, 0.0005, 0.01]
        step_length = [0.005, 0.005, 0.005, 0.005]
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
                    act = self.predict(obs)
                    obs, reward, done, _ = env.step(act)
                    obs = self.add_noisy(obs)
                    obs = clip(obs)
                    total_reward += reward
                    step_size += 1
                    if done:
                        break
                reward_list.append(step_size)
                # print('EVALUATE: ', i, step_size)
            print('AVG REWARD: ', np.mean(reward_list))
            iteration_mean_reward_list.append(np.array(reward_list))
            self.update_noisy()
        return np.array(iteration_mean_reward_list)


def clip(state):
    # low = state_space[0][0]
    # high = state_space[1][0]
    state_space1 = [[-4.79, -9.99, -0.419, -9.99], [4.79, 9.99, 0.419, 9.99]]
    state[0] = np.clip(state[0], state_space1[0][0], state_space1[1][0])
    state[1] = np.clip(state[1], state_space1[0][1], state_space1[1][1])
    state[2] = np.clip(state[2], state_space1[0][2], state_space1[1][2])
    state[3] = np.clip(state[3], state_space1[0][3], state_space1[1][3])
    return state


def train(agent, episode):
    agent.update_egreed()
    obs = env.reset()
    total_reward = 0
    ab_s = agent.divide_tool.get_abstract_state(obs)
    step_size = 0
    for t in range(TRAIN_MAX_STEP):  # 实际上t最大为500，超过500，gym就会将其置为done
        act = agent.sample(obs)  # 根据状态，选取动作(e贪心)
        next_obs, reward, done, _ = env.step(act)  # 环境执行动作
        next_obs = clip(next_obs)
        step_size += 1
        # if done and t < 499:
        #     reward = -1
        next_abs = agent.divide_tool.get_abstract_state(next_obs)
        # 转换为相应的数据结构，并放到经验回放盒
        trans = Transition(str_to_list(ab_s), act, reward, str_to_list(next_abs), done)
        agent.store_transition(trans)
        # 按照指定频率进行学习，更新网络参数
        if t % LEARN_FREQ == 0:
            agent.learn()
        obs = next_obs
        total_reward += reward
        if done:
            break
        ab_s = next_abs
    # print('episode',episode,"   total_reward", total_reward)
    return step_size


def evaluate(agent, episo):
    min_reward = EVALUTE_MAX_STEP
    min_step = EVALUTE_MAX_STEP

    for epi in range(episo):
        max_x = 0
        max_theta = 0
        total_reward = 0
        obs = env.reset()
        step_size = 0
        for i in range(EVALUTE_MAX_STEP):
            act = agent.predict(obs)
            obs, reward, done, _ = env.step(act)
            x = abs(obs[0])
            theta = abs(obs[2])
            max_x = max(max_x, x)
            max_theta = max(max_theta, theta)
            step_size += 1
            total_reward += reward
            if done:
                min_reward = min(total_reward, min_reward)
                min_step = min(min_step, i)
                print(epi, '#######DONE! step unfinished!  evaluate total_reward:', total_reward, max_x, max_theta, i)
                break
        print(epi, '  evaluate total_reward', total_reward, max_x, max_theta, step_size)
    return min_step


def train_model(agent):
    # agent = Agent(rtree)
    reward_list = []  # 初始化list，记录训练过程中的step
    mean_reward_list = []
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    for episode in range(EPISODES_NUM):
        reward = train(agent, episode)
        reward_list.append(reward)
        if episode % 50 == 49:  # 每训练XXX次，保存一下模型,并打印一下
            agent.save()
            print('训练回合数目=', episode + 1)
            m = np.mean(reward_list[-50:])
            print('avg reward:', m)
            mean_reward_list.append(m)
            # 先训练XXX步，然后看最近XXX步是否都是500,如果是的话，开始evaluate#并根据结果决定是否停止训练。
            if episode >= LEAST_EPISODES_NUM and m >= TRAIN_MAX_STEP:
                min = evaluate(agent, 10)
                if min >= EVALUTE_MAX_STEP:
                    agent.save()
                    print('达到训练终止条件：训练回合数目=', episode + 1)
                    return np.array(reward_list), np.array(mean_reward_list, dtype=np.float32)
    return np.array(reward_list), np.array(mean_reward_list, dtype=np.float32)
    # print('达到evaluate的条件：mean reward:', np.mean(reward_list[-100:]))
    # print('start evaluate:')
    # min_reward = evaluate(agent, 100)
    # print('evaluate result min_reward:', min_reward)
    # # 每次测试100个轨迹，要求均大于495，否则继续训练
    # if min_reward > EVALUTE_MAX_STEP-1:
    #     print('finish training...episode:', episode + 1, "   min_reward:", min_reward)
    #     agent.save()
    #     break





if __name__ == "__main__":
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 2], 'cart_abs1')

    agent = Agent(divide_tool)
    # agent.load()
    train_model(agent)
