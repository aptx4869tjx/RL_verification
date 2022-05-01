import torch
from scipy.optimize import minimize, Bounds
import math
import numpy as np

from verify.divide_tool import str_to_list


class PendulumEnv():

    def __init__(self, divide_tool, network):
        self.initial_state = [1, 0, 0.1]
        self.initial_state_region = None

        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 800
        self.atomic_propositions = ['safe']
        self.formula = 'not(A(G(safe)))'

        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.0
        self.m = 1.
        self.l = 1.

        self.action = 0.0
        self.epsilon = 0.0

        self.divide_tool = divide_tool
        self.network = network

    # state, inteval, range
    def is_done(self, s):

        done = bool(
            s[0] <= 0.0
        )
        return done

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

    def get_abstract_state_label(self, abstract_state, cnt):
        state_list = str_to_list(abstract_state)
        if state_list[0] <= 0:
            # if cnt <= 5:
            #     print('bad state-----------', state_list)
            return []
        return ['safe']

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

    # Define related functions for optimization in SciPy
    def thdot_minimum(self, x):
        # x[1] + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) *
        return x[1] + (3 * self.g / (2 * self.l) * math.sin(x[0]) + 3. / (self.m * self.l ** 2) * self.action) * self.dt

    def thdot_maximum(self, x):
        return - self.thdot_minimum(x)

    def cos_minimum(self, x):
        t = self.thdot_minimum(x)
        # cos(newth) = cos(th + t * dt) = cos(th) * cos(t * dt)  - sin(th) * sin(t * dt)
        cosin = math.cos(x[0] + t * self.dt)
        return cosin

    def cos_maximum(self, x):
        return - self.cos_minimum(x)

    def sin_minimum(self, x):
        t = self.thdot_minimum(x)
        # sin(newth) = sin(th + t * dt) = sin(th) * cos(t * dt)  + cos(th) * cos(t * dt)
        sine = math.sin(x[0] + t * self.dt)
        return sine

    def sin_maximum(self, x):
        return - self.sin_minimum(x)

    # Method must be implemented by users
    def get_next_states(self, current):
        cs = current
        try:
            current = str_to_list(current)
        except:
            print(current)
            exit(0)
        if current[0] <= 0:
            return [cs]

        s0 = torch.tensor(current, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()

        self.action = np.clip(action[0], -self.max_torque, self.max_torque)

        tmp1 = np.clip(current[1], -1, 1)
        tmp2 = np.clip(current[4], -1, 1)
        min_th = math.asin(tmp1)
        max_th = math.asin(tmp2)
        x0 = [min_th, current[2]]

        bounds = Bounds([min_th, current[2]], [max_th, current[5]])
        thdot_left = minimize(self.thdot_minimum, x0, method='SLSQP', bounds=bounds)
        thdot_left = self.thdot_minimum(thdot_left.x)
        thdot_right = minimize(self.thdot_maximum, x0, method='SLSQP', bounds=bounds)
        thdot_right = - self.thdot_maximum(thdot_right.x)

        cos_left = minimize(self.cos_minimum, x0, method='SLSQP', bounds=bounds)
        cos_left = self.cos_minimum(cos_left.x)
        cos_right = minimize(self.cos_maximum, x0, method='SLSQP', bounds=bounds)
        cos_right = - self.cos_maximum(cos_right.x)

        if cos_left < -1.0 or cos_right > 1.0 or cos_left > cos_right:
            print('pendulum/scenario_transition/cosin: ', current)
            print(cos_left, cos_right)
            exit(0)

        sin_left = minimize(self.sin_minimum, x0, method='SLSQP', bounds=bounds)
        sin_left = self.sin_minimum(sin_left.x)
        sin_right = minimize(self.sin_maximum, x0, method='SLSQP', bounds=bounds)
        sin_right = - self.sin_maximum(sin_right.x)

        if sin_left < -1.0 or sin_right > 1.0 or sin_left > sin_right:
            print('pendulum/scenario_transition/sine: ', current)
            print(sin_left, sin_right)
            exit(0)

        thdot_left = np.clip(thdot_left, -self.max_speed, self.max_speed)
        thdot_right = np.clip(thdot_right, -self.max_speed, self.max_speed)

        next_bounds = [cos_left, sin_left, thdot_left, cos_right, sin_right, thdot_right]
        next_states = self.divide_tool.intersection(next_bounds)
        return next_states

    def get_low_dim_state(self, state):
        s = str_to_list(state)
        res = [s[0], s[1], s[3], s[4]]
        obj_str = ','.join([str(_) for _ in res])
        return obj_str
