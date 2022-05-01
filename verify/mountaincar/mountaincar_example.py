import time

from abstract.mc.mountaincar_abs import Agent, state_space, train_model, evaluate
from verify.cegar import cegar
from verify.divide_tool import initiate_divide_tool_rtree
from verify.mountaincar.mountaincar_env import MountainCarTest

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
file_name = 'mc'
initial_intervals = [0.01, 0.001]
divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
agent = Agent(divide_tool)
mc = MountainCarTest(divide_tool, agent.network)
cegar(file_name, agent, divide_tool, train_model, mc, 4)
evaluate(agent)



