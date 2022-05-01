import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path


class ToraEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None

        high = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, u):
        p, v, th, thdot = self.state

        # print(p, v, th, thdot)

        u = np.clip(u, -2.0, 2.0)[0]

        t = 0.02
        p_new = p + v * t
        v_new = v + (-p + 0.1 * np.sin(th)) * t
        th_new = th + thdot * t
        thdot_new = thdot + u * t

        th_new = self.angle_normalize(th_new)

        self.state = np.array([p_new, v_new, th_new, thdot_new], dtype=np.float32)

        reward = 1.0
        done = bool(
            abs(p_new) > 5.0 or
            abs(v_new) > 5.0 or
            abs(th_new) > math.pi / 2.0 or
            abs(thdot_new) > 5.0
        )

        if done:
            reward = 0

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.5, 0.5, 1.0, 0.2])
        high = np.array([-0.75, -0.43, 0.54, -0.28])
        low = np.array([-0.77, -0.45, 0.51, -0.3])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
