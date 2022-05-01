from abstract.cartpole.cartpole_abs import *
from verify.cartpole.cart_env import CartPoleEnv
from verify.cegar import cegar
from verify.divide_tool import initiate_divide_tool_rtree


file_name = 'cart_abs2'
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
t0 = time.time()
initial_intervals = [0.16, 0.02, 0.016, 0.02]
divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 2], file_name)
agent = Agent(divide_tool)
# agent.load()
# train_model(agent)
cp = CartPoleEnv(divide_tool, agent.model)
cegar(file_name, agent, divide_tool, train_model, cp, 3)

