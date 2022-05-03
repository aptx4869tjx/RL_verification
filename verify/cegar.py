import time

import numpy as np
from rtree import index

from verify.Validator import Validator
from verify.divide_tool import initiate_divide_tool_rtree

record_num = 1


def cegar(file_name, agent, divide_tool, train_model, verify_env, max_iteration):
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()

    # agent.load()
    train_model(agent)

    agent.load()
    # evaluate(agent)
    tr = time.time()

    v = Validator(verify_env, agent.network)
    k = v.create_kripke_ctl()
    t1 = time.time()
    if k is None:
        violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
        res2 = True
        print('number of counterexamples：', len(violated_states))
    else:
        # v.formula = 'not(A(G(safe)))'
        res2, violated_states = v.ctl_model_check(k)
        print('number of counterexamples：', len(violated_states))
    t2 = time.time()
    print('train time:', tr - t0, 'construct kripke structure:', t1 - tr, 'model checking:', t2 - t1)
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    start_id = divide_tool.rtree.get_size()
    p = index.Property()
    p.dimension = len(divide_tool.key_dim)
    iteration_time = 1
    while res2 and iteration_time <= max_iteration:
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        t0 = time.time()
        divide_tool.rtree = index.Index(file_name, divide_tool.rtree_refinement(violated_states, start_id),
                                        properties=p)
        print('number of states after refinement:', divide_tool.rtree.get_size())
        trr = time.time()
        start_id += divide_tool.rtree.get_size()
        agent.divide_tool = divide_tool
        # agent.load()
        agent.reset()
        train_model(agent)

        agent.load()
        # evaluate(agent)
        tr = time.time()
        verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        # pd = PendulumEnv(divide_tool, agent.actor)
        v = Validator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            print('number of counterexamples：', len(violated_states))
        print('iteration :', iteration_time, res2, len(violated_states))
        t2 = time.time()
        print('refine:', trr - t0, 'train:', tr - trr, 'construct kripke structure:', t1 - tr, 'model checking:',
              t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        iteration_time += 1
    while res2:
        t0 = time.time()
        # agent.load()
        agent.reset()
        train_model(agent)

        agent.load()
        tr = time.time()
        # verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        v = Validator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            print('number of counterexamples：', len(violated_states))
        t2 = time.time()
        print('train:', tr - t0, 'construct kripke structure:', t1 - tr, 'model checking:', t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))


def cegar_record(file_name, agent, divide_tool, train_model, verify_env, max_iteration):
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    iteration_reward_list = []
    iteration_mean_reward_list = []
    iteration_time = 0
    # agent.load()
    iteration_mean_reward_list.append([])
    for i in range(record_num):
        print('iteration: ', iteration_time, 'cycle: ', i)
        agent.reset()
        reward_list, mean_reward_list = train_model(agent)
        iteration_mean_reward_list[iteration_time].append(mean_reward_list)
    agent.load()
    # evaluate(agent)
    tr = time.time()

    v = Validator(verify_env, agent.network)
    k = v.create_kripke_ctl()
    t1 = time.time()
    if k is None:
        violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
        res2 = True
        print('number of counterexamples：', len(violated_states))
    else:
        # v.formula = 'not(A(G(safe)))'
        res2, violated_states = v.ctl_model_check(k)
    t2 = time.time()
    print(tr - t0, t1 - tr, t2 - t1)
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    start_id = divide_tool.rtree.get_size()
    p = index.Property()
    p.dimension = len(divide_tool.key_dim)

    while res2 and iteration_time < max_iteration:
        iteration_time += 1
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        t0 = time.time()
        divide_tool.rtree = index.Index(file_name, divide_tool.rtree_refinement(violated_states, start_id),
                                        properties=p)
        print('精化后状态数量:', divide_tool.rtree.get_size())
        trr = time.time()
        start_id += divide_tool.rtree.get_size()
        agent.divide_tool = divide_tool
        # agent.load()
        iteration_mean_reward_list.append([])
        for i in range(record_num):
            print('iteration: ', iteration_time, 'cycle: ', i)
            agent.reset()
            reward_list, mean_reward_list = train_model(agent)
            iteration_mean_reward_list[iteration_time].append(mean_reward_list)
        save_file(file_name, iteration_mean_reward_list)
        agent.load()
        # evaluate(agent)
        tr = time.time()
        verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        # pd = PendulumEnv(divide_tool, agent.actor)
        v = Validator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
        print('iteration :', iteration_time, res2, len(violated_states))
        t2 = time.time()
        print(trr - t0, tr - trr, t1 - tr, t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    while res2:
        iteration_time += 1
        t0 = time.time()
        # agent.load()
        iteration_mean_reward_list.append([])
        for i in range(record_num):
            print('iteration: ', iteration_time, 'cycle: ', i)
            # agent.reset()
            agent.load()
            reward_list, mean_reward_list = train_model(agent)
            iteration_mean_reward_list[iteration_time].append(mean_reward_list)
        save_file(file_name, iteration_mean_reward_list)
        agent.load()
        tr = time.time()
        # verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        v = Validator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
        t2 = time.time()
        print(tr - t0, t1 - tr, t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    # np.save('R4' + file_name + '.npy', arr=np.array(iteration_reward_list))
    save_file(file_name, iteration_mean_reward_list)
    res_list = robust_analysis(agent, train_model, verify_env)
    np.save('Robust3' + file_name + '.npy', arr=res_list)
    f1 = open('R3' + file_name + ".txt", "w")
    f1.writelines(str(res_list))
    f1.close()


def robust_analysis(agent, train_model, verify_env):
    iteration_avg_reward_list = []
    for i in range(20):
        print('train iteration : ', i)
        avg_reward_list = agent.evaluate_with_noisy(30)
        iteration_avg_reward_list.append(avg_reward_list)
        res2 = True
        train_num = 0
        while res2:
            agent.reset()
            # agent.load()
            t0 = time.time()
            train_model(agent)
            train_num += 1
            agent.load()
            tr = time.time()
            # verify_env.divide_tool = divide_tool
            verify_env.network = agent.network
            v = Validator(verify_env, agent.network)
            k = v.create_kripke_ctl()
            t1 = time.time()
            if k is None:
                # violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
                res2 = True
                # print('反例数量：', len(violated_states))
            else:
                # v.formula = 'not(A(G(safe)))'
                res2, violated_states = v.ctl_model_check(k)
            t2 = time.time()
            print('train iteration : ', i, 'train num :', train_num)
            print(tr - t0, t1 - tr, t2 - t1)
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    return np.array(iteration_avg_reward_list)


def save_file(file_name, iteration_mean_reward_list):
    np.save('RMean1' + file_name + '.npy', arr=np.array(iteration_mean_reward_list))
    f = open('R' + file_name + ".txt", "w")
    f.writelines(str(iteration_mean_reward_list))
    f.close()
