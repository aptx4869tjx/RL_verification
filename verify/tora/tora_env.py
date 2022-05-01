import torch
from scipy.optimize import minimize, Bounds
import math
import numpy as np

from verify.divide_tool import str_to_list


class Tora_Env():

    def __init__(self, divide_tool, network):
        self.initial_state = [-0.76, -0.44, 0.52, -0.29]
        self.initial_state_region = [-0.77, -0.45, 0.51, -0.3, -0.75, -0.43, 0.54, -0.28]

        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 20
        self.atomic_propositions = ['safe']

        self.formula = 'not(A(G(safe)))'
        self.th = 2.0

        self.action = 0.0

        self.divide_tool = divide_tool
        self.network = network

    # state, inteval, range
    def is_done(self, s):
        done = bool(
            abs(s[0]) > 1.5 or abs(s[1]) > 1.5
        )
        return done

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

    def get_abstract_state_label(self, abstract_state, cnt):
        state_list = str_to_list(abstract_state)
        if state_list[0] >= -1.5 and state_list[2] <= 1.5 and state_list[1] >= -math.pi / 2.0 and state_list[
            3] <= math.pi / 2.0:
            return ['safe']
        return []

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def next_abstract_domain(self, abstract_obs, act):
        deltat = 0.02
        # p0, p1, v0, v1, th0, th1, thdot0, thdot1 = abstract_obs
        p0, v0, th0, thdot0, p1, v1, th1, thdot1 = abstract_obs
        if th0 < -np.pi / 2 and th1 > -np.pi / 2:
            sin_math0 = -1.0
            sin_math1 = max(math.sin(th0), math.sin(th1))
        elif th0 < np.pi / 2 and th1 > np.pi / 2:
            sin_math1 = 1.0
            sin_math0 = min(math.sin(th0), math.sin(th1))
        else:
            sin_math0 = min(np.sin(th0), np.sin(th1))
            sin_math1 = max(np.sin(th0), np.sin(th1))

        p0_new = p0 + v0 * deltat
        p1_new = p1 + v1 * deltat
        th0_new = th0 + thdot0 * deltat
        th1_new = th1 + thdot1 * deltat
        thdot0_new = thdot0 + act * deltat
        thdot1_new = thdot1 + act * deltat
        v0_new = v0 + (min(-p0, -p1) + 0.1 * sin_math0) * deltat
        v1_new = v1 + (max(-p0, -p1) + 0.1 * sin_math1) * deltat

        p0_new = np.clip(p0_new, -10.0, 10.0)
        p1_new = np.clip(p1_new, -10.0, 10.0)
        v0_new = np.clip(v0_new, -10.0, 10.0)
        v1_new = np.clip(v1_new, -10.0, 10.0)
        thdot0_new = np.clip(thdot0_new, -10.0, 10.0)
        thdot1_new = np.clip(thdot1_new, -10.0, 10.0)

        th0_new = self.angle_normalize(th0_new)
        th1_new = self.angle_normalize(th1_new)
        real_th0_new = min(th0_new, th1_new)
        real_th1_new = max(th0_new, th1_new)
        return [p0_new, p1_new, v0_new, v1_new, real_th0_new, real_th1_new, thdot0_new, thdot1_new]

    # Method must be implemented by users
    def get_next_states(self, current):
        cs = current
        try:
            current = str_to_list(current)
        except:
            print(current)
            exit(0)
        if current[0] < -1.5 or current[4] > 1.5 or current[1] < -math.pi / 2.0 or current[5] > math.pi / 2.0:
            return [cs]

        s0 = torch.tensor(current, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()
        self.action = np.clip(action[0], -self.th, self.th)

        next_bounds = self.next_abstract_domain(current, self.action)
        next_bounds = convert(next_bounds)

        next_states = self.divide_tool.intersection(next_bounds)
        return next_states

    def get_low_dim_state(self, state):
        s = str_to_list(state)
        res = [s[0], s[1], s[4], s[5]]
        obj_str = ','.join([str(_) for _ in res])
        return obj_str


def convert(s):
    for i in range(len(s)):
        s[i] = np.clip(s[i], -10, 10)

    return [s[0], s[2], s[4], s[6], s[1], s[3], s[5], s[7]]
