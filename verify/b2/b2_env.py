import torch
from scipy.optimize import minimize, Bounds
import math
import numpy as np

from verify.divide_tool import str_to_list


class B2_Env():

    def __init__(self, divide_tool, network):
        self.initial_state = [0.7001, 0.7001]
        self.initial_state_region = [0.7, 0.7, 0.9, 0.9]

        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 800
        self.atomic_propositions = ['goal', 'safe']
        # self.formula = 'not(A(safe U goal) or A(G(safe)))'
        self.formula= 'not(A(F(goal)))'
        self.th = 1.0

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
        try:
            if state_list[0] >= -0.3 and state_list[2] <= 0.1 and state_list[1] >= -0.35 and state_list[3] <= 0.5:
                return ['goal', 'safe']
        except:
            print(state_list)
        if state_list[0] >= -1.5 and state_list[2] <= 1.5 and state_list[1] >= -1.5 and state_list[3] <= 1.5:
            return ['safe']
        return []

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

    # Method must be implemented by users
    def get_next_states(self, current):
        cs = current
        try:
            current = str_to_list(current)
        except:
            print(current)
            exit(0)
        if current[0] >= -0.3 and current[2] <= 0.1 and current[1] >= -0.35 and current[3] <= 0.5:
            return [cs]

        s0 = torch.tensor(current, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()
        t = 0.2
        self.action = np.clip(action[0], -self.th, self.th)
        pl = current[0]
        vl = current[1]
        pr = current[2]
        vr = current[3]
        # pl,vl,pr,vr
        # v_new = v + (u * v * v - p) * t
        p_l = pl + (vl - pr * pr * pr) * t
        p_r = pr + (vr - pl * pl * pl) * t
        v_l = vl + self.action * t
        v_r = vr + self.action * t
        p_l = np.clip(p_l, -2, 2)
        p_r = np.clip(p_r, -2, 2)
        v_l = np.clip(v_l, -2, 2)
        v_r = np.clip(v_r, -2, 2)

        next_bounds = [p_l, v_l, p_r, v_r]
        next_states = self.divide_tool.intersection(next_bounds)
        return next_states

    def get_low_dim_state(self, state):
        return state
        # s = str_to_list(state)
        # res = [s[0], s[1], s[3], s[4]]
        # obj_str = ','.join([str(_) for _ in res])
        # return obj_str
