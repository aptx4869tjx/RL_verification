#!/usr/bin/python
# coding=utf-8
import copy
import os
import random
import sys
import time
from collections import deque, namedtuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rtree import index
from scipy.optimize import Bounds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# 获取文件所在的当前路径
from verify.divide_tool import str_to_list, initiate_divide_tool_rtree

script_path = os.path.split(os.path.realpath(__file__))[0]
# 判断是否使用double DQN
USE_DBQN = False
# 生成需要保存的文件名称以及路径
pt_file = os.path.join(script_path, "mountaincar-dqn.pt")

# 创建测试环境testbed
env = gym.make("MountainCar-v0")
# 创建需要保存的数据结构
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)
OBS_N = 4  # 整型，状态空间维数
ACT_N = env.action_space.n  # 整型，动作空间个数
HIDDEN_N = 64  # 隐藏层节点数目
MEMORY_CAPACITY = 10000  # 经验回放盒大小
WARMUP_SIZE = 256  # 经验回放盒大于该数目，才会开始采样学习，否则不采样
BATCH_SIZE = 128  # 每次采样数目
MODEL_SYNC_COUNT = 20  # 目标网络的更新频率
LEARNING_RATE = 1e-3  # 学习率
LEARN_FREQ = 8  # 并不是每次都学习，每8次才会学习一次
WEIGHT_DECAY = 0
GAMMA = 0.99  # 累计衰减系数
E_GREED = 0.1  # 贪心算法的权重e
E_GREED_DEC = 1e-5  # e每次下降的次数
E_GREED_MIN = 0.01  # e的最小值
EPISODES_NUM = 20000  # 回合数目


class Model(nn.Module):  # 神经网络的类
    def __init__(self):  # 网络结构及其初始化
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
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
        self.target_model = Model()  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.network = Model()
        self.memory = deque(maxlen=MEMORY_CAPACITY)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数

        self.e_greed = E_GREED
        self.update_count = 0  # 更新次数
        self.divide_tool = divide_tool

    def reset(self):
        self.model = Model()  # 当前网络
        self.target_model = Model()  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.network = Model()
        self.memory = deque(maxlen=MEMORY_CAPACITY)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数

        self.e_greed = E_GREED
        self.update_count = 0  # 更新次数

    def update_egreed(self):  # 更新贪心参数e
        self.e_greed = max(E_GREED_MIN, self.e_greed - E_GREED_DEC)

    def predict(self, obs):  # 选取最大的action
        abs = str_to_list(self.divide_tool.get_abstract_state(obs))
        q_val = self.model(torch.FloatTensor(abs)).detach().numpy()
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return np.random.choice(choice_list)

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
        self.network.load_state_dict(self.model.state_dict())
        self.target_model.load_state_dict(self.model.state_dict())
        print(pt_file + " loaded.")

    def evaluate(self, epi):
        min_reward = 200
        reward_list = []
        for t in range(epi):
            total_reward = 0
            obs = env.reset()
            while True:
                act = self.predict(obs)
                obs, reward, done, _ = env.step(act)
                total_reward += reward
                if done:
                    # print(total_reward)
                    reward_list.append(total_reward)
                    if total_reward < min_reward:
                        min_reward = total_reward
                    # print('test', t, total_reward)
                    break
        print('avg: ----', np.mean(reward_list))

        # f = open("mountaincar_reward.txt", "w")
        #
        # f.writelines(str(reward_list))
        # f.close()
        # a = np.array(a)
        np.save('mountaincar_reward.npy', np.array(reward_list))  # 保存为.npy格式

        return min_reward


def train(agent):
    agent.update_egreed()
    # print(agent.rtree.get_size())
    # print(agent.e_greed)
    obs = env.reset()
    total_reward = 0
    step_size = 0
    ab_s = agent.divide_tool.get_abstract_state(obs)
    for t in range(10000):  # 实际上t最大为200，超过200，gym就会将其置为done
        # act = agent.sample(obs)  # 根据状态，选取动作(e贪心)
        act = agent.sample(obs)
        # if np.random.rand() < agent.e_greed:
        #     act = np.random.choice(ACT_N)
        # else:
        #     q_val = agent.model(torch.FloatTensor(change_pos_vel(abstract_state_to_list(ab_s)))).detach().numpy()
        #     q_max = np.max(q_val)
        #     choice_list = np.where(q_val == q_max)[0]
        #     act = np.random.choice(choice_list)

        next_obs, reward, done, _ = env.step(act)  # 环境执行动作
        next_abs = agent.divide_tool.get_abstract_state(next_obs)
        step_size += 1
        # if next_obs[0] > 0.3:
        #     reward += 1
        # if next_obs[1] > 0.5:
        #     reward += 1
        # 转换为相应的数据结构，并放到经验回放盒
        trans = Transition(str_to_list(ab_s), act, reward,
                           str_to_list(next_abs), done)
        agent.store_transition(trans)

        # 按照指定频率进行学习，更新网络参数
        if t % LEARN_FREQ == 0:
            agent.learn()

        obs = next_obs
        total_reward += reward
        if done:
            break
        ab_s = next_abs
    # print("total_reward", total_reward, 'step_size', step_size)
    return total_reward


def print_avg_reward(agent):
    reward_list = []
    for t in range(100):
        total_reward = 0
        obs = env.reset()
        while True:
            act = agent.predict(obs)
            obs, reward, done, _ = env.step(act)
            total_reward += reward
            if done:
                reward_list.append(total_reward)
                break
    print('精华后的平均reward: ', np.mean(reward_list))


def evaluate(agent, epi=100):
    min_reward = 200
    reward_list = []
    for t in range(epi):
        total_reward = 0
        obs = env.reset()
        while True:
            act = agent.predict(obs)
            obs, reward, done, _ = env.step(act)
            total_reward += reward
            if done:
                # print(total_reward)
                reward_list.append(total_reward)
                if total_reward < min_reward:
                    min_reward = total_reward
                # print('test', t, total_reward)
                break
    print('avg: ----', np.mean(reward_list))
    return min_reward


# [0.01, 0.001] 80 0000
state_space = [[-1.4, -0.09], [0.8, 0.09]]
initial_intervals = [0.01, 0.001]


# def main():
#     rindex = '3'
#     p = index.Property()
#     print(len(state_space.lb))
#     p.dimension = len(state_space.lb)
#
#     rtree = index.Index('mountaincar-rtree' + rindex, properties=p)
#     print(rtree.get_size())
#     t1 = time.time()
#     if rtree.get_size() == 0:
#         low_bound = copy.copy(state_space.lb)
#         up_bound = copy.copy(state_space.ub)
#         rtree = index.Index('mountaincar-rtree' + rindex, divide(0, low_bound, up_bound), properties=p)
#         # divide(0, low_bound, up_bound)
#         # 判断各个维度是否已经全部划分完毕
#         print('\n', rtree.get_size())
#     t2 = time.time()
#     print(t2 - t1)
#     agent = Agent(rtree)
#
#     if os.path.exists(pt_file):
#         agent.load()
#
#     # reward = evaluate(agent)
#     reward_list = []
#     t0 = time.time()
#     for episode in range(EPISODES_NUM):
#         ep_reward = train(agent)
#         reward_list.append(ep_reward)
#         if episode % 100 == 99:
#             t1 = time.time()
#             print('episode', episode, np.mean(reward_list[-100:]), 'time', t1 - t0)
#             t0 = t1
#
#         if episode >= 5000 and episode % 500 == 499:
#             min_reward = evaluate(agent)
#             if min_reward > -130:
#                 print('finish training...episode:', episode, "   min_reward:", min_reward)
#                 agent.save()
#                 break
#             else:
#                 print('continue train....    min reward =', min_reward)
#
#         if episode % 50 == 49:
#             agent.save()
#
#     evaluate(agent)


def train_model(agent):
    t0 = time.time()
    reward_list = []
    for episode in range(EPISODES_NUM):
        ep_reward = train(agent)
        reward_list.append(ep_reward)
        if episode % 100 == 99:
            t1 = time.time()
            print('episode', episode, np.mean(reward_list[-100:]), 'time', t1 - t0)
            t0 = t1
        if episode >= 5000 and episode % 500 == 0:
            min_reward = evaluate(agent, 100)
            if min_reward > -130:
                print('finish training...episode:', episode, "   min_reward:", min_reward)
                agent.save()
                break
            else:
                print('continue train....    min reward =', min_reward)
        if episode % 50 == 49:
            agent.save()


if __name__ == "__main__":
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], 'mou_rtree')
    agent = Agent(divide_tool)
    agent.load()
    # train_model(agent)
    # min_reward = agent.evaluate(100)
    min_reward = evaluate(agent, 100)

    # a = np.load('mountaincar_reward.npy')
    # a = a.tolist()
    # print('aaa')
