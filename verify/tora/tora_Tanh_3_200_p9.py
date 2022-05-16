import time

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from abstract.tora.tora_abs2 import *
from verify.cegar import cegar, cegar_record
from verify.divide_tool import initiate_divide_tool_rtree, initiate_divide_tool
from verify.tora.tora_env import Tora_Env

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
file_name = 'tora_Tanh_3_200_p9'
initial_intervals = [0.1, 0.1, 0.1, 0.1]
divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
agent = Agent(divide_tool)
tora = Tora_Env(divide_tool, agent.actor)
cegar(file_name, agent, divide_tool, train_model, tora, 4)
evaluate(agent)
