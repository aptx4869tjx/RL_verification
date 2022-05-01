import math
import time

import numpy as np
import torch
from rtree import index
from scipy.optimize import Bounds, minimize


class MountainCarTest:

    def __init__(self, divide_tool, network):
        self.initial_state = [-0.5, 0]

        self.initial_state_region = None
        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 3000
        self.atomic_propositions = ['success']
        self.formula = 'not(A(F(success)))'

        self.divide_tool = divide_tool

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0

        self.force = 0.001
        self.gravity = 0.0025
        # shared action
        self.g_action = 0

        self.network = network

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

    def string_to_list(self, ss):
        index = ss.find('i')
        s = ss[index + 1:].replace(':', ',')
        try:
            list(map(float, s.split(',')))
        except:
            print(ss, '----', s)
            exit(0)
        return list(map(float, s.split(',')))

    # def change_pos_vel(self, s_list):
    #     pmin = s_list[0]
    #     vmin = s_list[1]
    #     pmax = s_list[2]
    #     vmax = s_list[3]
    #     return [pmin, pmax, vmin, vmax]

    def get_abstract_state_label(self, abstract_state, cnt):
        abstract_state = self.string_to_list(abstract_state)
        if abstract_state[0] >= 0.5:
            return ['success']
        return []

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

    def is_done(self, current):
        return self.string_to_list(current)[0] >= 0.5
        # Define related functions for optimization in SciPy

    def next_abstract_domain(self, abstract_obs, act):
        # a = abstract_obs[0]
        # b = abstract_obs[1]
        # c = abstract_obs[2]
        # d = abstract_obs[3]
        a = abstract_obs[0]
        b = abstract_obs[2]
        c = abstract_obs[1]
        d = abstract_obs[3]
        if a <= 0 and b >= 0:
            temp_min = (act - 1) * 0.001 - math.cos(3 * 0) * 0.0025
            temp_max = min((act - 1) * 0.001 - math.cos(3 * a) * 0.0025, (act - 1) * 0.001 - math.cos(3 * b) * 0.0025)
        else:
            temp_max = max((act - 1) * 0.001 - math.cos(3 * a) * 0.0025, (act - 1) * 0.001 - math.cos(3 * b) * 0.0025)
            temp_min = min((act - 1) * 0.001 - math.cos(3 * a) * 0.0025, (act - 1) * 0.001 - math.cos(3 * b) * 0.0025)

        v_max = d + temp_max
        v_min = c + temp_min
        v_max = np.clip(v_max, -0.07, 0.07)
        v_min = np.clip(v_min, -0.07, 0.07)
        p_max = b + v_max
        p_min = a + v_min
        p_max = np.clip(p_max, -1.2, 0.6)
        p_min = np.clip(p_min, -1.2, 0.6)
        return p_min, v_min, p_max, v_max

    def velocity_minimum(self, x):
        return x[1] + (self.g_action - 1) * self.force + math.cos(3 * x[0]) * (-self.gravity)

    def velocity_maximum(self, x):
        return - self.velocity_minimum(x)

    def get_next_states(self, current):
        cs = current
        try:
            current = self.string_to_list(current)
        except:
            print(current)
            exit(0)
        if current[0] >= 0.5:
            return [cs]

        out = self.network(torch.Tensor(current)).detach()  ##detch()截断反向传播的梯度
        self.g_action = torch.argmax(out).data.item()
        next_bounds = self.next_abstract_domain(current, self.g_action)
        # if g_action == 2:
        #     print('lllll')

        # bounds = Bounds(current[0:2], current[2:4])
        # x0 = [(current[0] + current[2]) / 2, (current[1] + current[3]) / 2]
        # t1 = time.time()
        # vel_left = minimize(self.velocity_minimum, x0, method='TNC', bounds=bounds)
        # vel_left = self.velocity_minimum(vel_left.x)
        # vel_right = minimize(self.velocity_maximum, x0, method='TNC', bounds=bounds)
        # vel_right = - self.velocity_maximum(vel_right.x)
        #
        # t2 = time.time()
        # # print((t2 - t1) * 1000)
        # vel_left = np.clip(vel_left, -self.max_speed, self.max_speed)
        # vel_right = np.clip(vel_right, -self.max_speed, self.max_speed)

        # pos_left = current[0] + vel_left
        # pos_right = current[2] + vel_right
        #
        # pos_left = np.clip(pos_left, self.min_position, self.max_position)
        # pos_right = np.clip(pos_right, self.min_position, self.max_position)
        #
        # if pos_left <= self.min_position and vel_left < 0:
        #     vel_left = 0
        #     vel_right = max(0, vel_right)
        # error = 0
        # next_bounds = [pos_left + error, vel_left + error, pos_right - error, vel_right - error]
        # print(pos_left)
        # t3 = time.time()
        # try:
        # print('nextbounds',next_bounds)
        # print(next_bounds)
        next_states = self.divide_tool.intersection(next_bounds)

        # t4 = time.time()
        # print(t2 - t1, t4 - t3)
        return next_states

    def get_low_dim_state(self, state):
        return state
