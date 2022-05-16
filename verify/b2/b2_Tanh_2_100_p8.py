import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from abstract.b2.b2_abs2 import state_space, Agent, train_model, initial_intervals, evaluate
from verify.Validator import Validator

from verify.b2.b2_env import B2_Env
from verify.cegar import cegar, cegar_record
from verify.divide_tool import initiate_divide_tool_rtree, initiate_divide_tool


print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
file_name = 'b2_Tanh_2_100_p8'
initial_intervals = [0.01, 0.01]
divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
agent = Agent(divide_tool)
b2 = B2_Env(divide_tool, agent.actor)
b2.formula = 'not(A(safe U goal) or A(G(safe)))'
cegar(file_name, agent, divide_tool, train_model, b2, 5)
evaluate(agent)

