import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from abstract.pendulum.pendulum_abs import *
from verify.cegar import cegar
from verify.pendulum.pendulum_env import PendulumEnv

# def cegar(file_name, agent, state_space, initial_intervals, train_model, verify_env, max_iteration):
file_name = 'pendulum_abstraction2'
initial_intervals = [0.16, 0.16, 0.01]
divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
agent = Agent(divide_tool)
pd = PendulumEnv(divide_tool, agent.actor)
cegar(file_name, agent, divide_tool, train_model, pd, 5)
