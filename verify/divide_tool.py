import bisect
import copy

import numpy as np
from rtree import index


# 用于生成不带rtree的divide_tool
def initiate_divide_tool(state_space, initial_intervals):
    divide_point = []
    for i in range(len(state_space[0])):
        lb = state_space[0][i]
        ub = state_space[1][i]
        tmp = [lb]
        while lb < ub:
            lb = round(lb + initial_intervals[i], 10)
            tmp.append(lb)
        divide_point.append(tmp)
    return DivideTool(divide_point)


# 用于生成带有rtree的divide_tool
def initiate_divide_tool_rtree(state_space, initial_intervals, key_dim, file_name):
    divide_point = []
    key_state_space = [[], []]
    key_initial_intevals = []
    for i in range(len(state_space[0])):
        if i in key_dim:
            key_state_space[0].append(state_space[0][i])
            key_state_space[1].append(state_space[1][i])
            key_initial_intevals.append(initial_intervals[i])
            continue
        lb = state_space[0][i]
        ub = state_space[1][i]
        tmp = [lb]
        while lb < ub:
            lb = round(lb + initial_intervals[i], 10)
            tmp.append(lb)
        divide_point.append(tmp)
    p = index.Property()
    p.dimension = len(key_dim)

    rtree = index.Index(file_name, properties=p)
    if rtree.get_size() == 0:
        rtree = index.Index(file_name, divide(key_state_space, key_initial_intevals), properties=p)
    print('number of states in rtree', rtree.get_size())
    dp = DivideTool(divide_point)
    dp.key_dim = key_dim
    dp.rtree = rtree
    return dp


# 用于yield rtree构造中的状态的上下界
def divide(state_space, intervals):
    lb = state_space[0]
    ub = state_space[1]
    np_initial_intervals = np.array(intervals)
    id = 1
    while True:
        flag = True
        for j in range(len(lb)):
            if lb[j] >= ub[j]:
                flag = False
                break
        if flag:
            upper = np.array(lb) + np_initial_intervals
            for i in range(len(upper)):
                upper[i] = round(upper[i], 10)
            # print(str(lb) + str(upper))
            # print(id)
            obj_str = ','.join([str(_) for _ in lb]) + ',' + ','.join([str(_) for _ in upper])
            # print(obj_str)
            yield id, tuple(lb) + tuple(upper.tolist()), obj_str
            # rtree.insert(id, tuple(lb) + tuple(upper.tolist()),
            #              obj=obj_str)
            id += 1
            if id % 1000 == 0:
                print(id)
        i, lb = get_next_lb(lb, state_space, intervals)
        # print(lb)
        if lb is None:
            break


# 给定一个当前的下界lb，以及状态范围和划分粒度，返回下一个lb
def get_next_lb(lb, state_space, intervals):
    tmp_lb = copy.copy(lb)
    flag = False
    for i in range(len(tmp_lb)):
        if tmp_lb[i] < state_space[1][i]:
            flag = True
            tmp_lb[i] = round(intervals[i] + tmp_lb[i], 10)
            if i >= 1:
                j = i - 1
                while j >= 0:
                    tmp_lb[j] = state_space[0][j]
                    j -= 1
            # if i >= 2:
            #     tmp_lb[1] = state_space.lb[1]
            # if i >= 3:
            #     tmp_lb[2] = state_space.lb[2]
            return i, tmp_lb
    if not flag:
        return -1, None


# 将一个bound的各个维度进行二分
def bound_bisect(bound):
    dim = int(len(bound) / 2)
    bounds = []
    interval = []
    state_space = [[], []]
    for i in range(dim):
        interval.append(round((bound[i + dim] - bound[i]) / 2, 10))
        state_space[0].append(bound[i])
        state_space[1].append(bound[i + dim])
    lb = bound[0:dim]

    count = 0
    # print(bound)
    while lb is not None:
        upper = np.array(lb) + np.array(interval)
        flag = True
        for i in range(len(upper)):
            upper[i] = round(upper[i], 10)
            if upper[i] > bound[i + dim]:
                flag = False
                break
        if flag:
            bounds.append(lb + upper.tolist())
        # print(lb)
        id1, lb = get_next_lb(lb, state_space, interval)
        count += 1
    return bounds


def str_to_list(state_str):
    return list(map(float, state_str.split(',')))


def list_to_str(state_list):
    res_list = []
    for tmp in state_list:
        obj_str = ','.join([str(_) for _ in tmp])
        res_list.append(obj_str)
    return res_list


class DivideTool:
    def __init__(self, divide_point):
        self.divide_point = divide_point
        self.key_dim = []
        self.rtree = None
        # self.part_state1 = None
        # self.part_state2 = None

    # 将给定的状态范围，分成普通的状态空间和关键状态空间
    def part_state(self, bound):
        dim = len(bound)
        half_dim = int(dim / 2)
        state_space = list(range(dim - 2 * len(self.key_dim)))
        key_state_space = list(range(2 * len(self.key_dim)))
        half_dim1 = int(len(state_space) / 2)
        half_dim2 = int(len(key_state_space) / 2)
        id1 = 0
        id2 = 0
        for i in range(half_dim):
            if i in self.key_dim:
                key_state_space[id2] = bound[i]
                key_state_space[id2 + half_dim2] = bound[i + half_dim]
                id2 += 1
            else:
                state_space[id1] = bound[i]
                state_space[id1 + half_dim1] = bound[i + half_dim]
                id1 += 1
        return state_space, key_state_space

    # 给定范围查询
    def intersection(self, bound):
        dim = len(bound)
        half_dim = int(dim / 2)
        state_space, key_state_space = self.part_state(bound)
        half_dim1 = int(len(state_space) / 2)
        half_dim2 = int(len(key_state_space) / 2)

        # 获取各个维度的划分点
        region = self.get_abstract_region(state_space)
        state_list = []
        abstract_state = list(range(len(state_space)))
        self.point_to_state(region, 0, abstract_state, state_list)
        length1 = len(state_list)

        res_list = []
        # 关键维度上的查询
        if self.rtree is not None:
            rtree_state_list = list(self.rtree.intersection(key_state_space, objects=True))
            length2 = len(rtree_state_list)

            final_state_list = []
            # for i in range(len(state_space)):

            for i in range(length1):
                part_state1 = state_list[i]
                for j in range(length2):
                    ik1 = 0
                    ik2 = 0
                    final_abs = list(range(dim))
                    part_state2 = rtree_state_list[j]
                    for k in range(half_dim):
                        if k in self.key_dim:
                            final_abs[k] = part_state2.bbox[ik2]
                            final_abs[k + half_dim] = part_state2.bbox[ik2 + half_dim2]
                            ik2 += 1
                        else:
                            final_abs[k] = part_state1[ik1]
                            final_abs[k + half_dim] = part_state1[ik1 + half_dim1]
                            ik1 += 1
                    final_state_list.append(final_abs)
            res_list = list_to_str(final_state_list)
        else:
            res_list = list_to_str(state_list)
        return res_list

    # 给定一个范围，返回在相应维度上的分界点
    def get_abstract_region(self, s):
        # s = s.tolist()
        dim = int(len(s) / 2)
        tmp = []

        # 范围查询
        for i in range(dim):
            tmp2 = []
            pos_left = bisect.bisect_left(self.divide_point[i], s[i])
            pos_right = bisect.bisect(self.divide_point[i], s[i + dim])
            if s[i] != self.divide_point[i][pos_left]:
                pos_left -= 1
            while pos_left <= pos_right:
                tmp2.append(self.divide_point[i][pos_left])
                pos_left += 1
            tmp.append(tmp2)
            # tmp[i] = divide_point[i][pos_left - 1]
            # tmp[i + dim] = divide_point[i][pos_right]
        return tmp

    # 具体状态到抽象状态，即查询具体状态
    def get_abstract_state(self, s):
        dim = len(s)
        # s需要是list
        # s = s.tolist()
        if type(s) is np.ndarray:
            s = s.tolist()
        bound = s + s
        s1, s2 = self.part_state(bound)
        tt = self.intersection(bound)
        if len(tt) == 0:
            print('no corresponding abstract state')
        return tt[0]
        # tmp = []
        # # 点查询
        # if dim == len(self.divide_point):
        #     tmp = list(range(2 * dim))
        #     for i in range(dim):
        #         pos = bisect.bisect(self.divide_point[i], s[i])
        #         tmp[i] = self.divide_point[i][pos - 1]
        #         tmp[i + dim] = self.divide_point[i][pos]
        #     obj_str = ','.join([str(_) for _ in tmp])
        #     return obj_str
        # return None

    # 将划分边界点，转化为抽象状态list
    def point_to_state(self, divide_point, current_dim, abstract_state, state_list):
        dim = len(divide_point)
        # abstract_state = list(range(2 * dim))

        if current_dim < dim:
            for i in range(len(divide_point[current_dim]) - 1):
                a_s = copy.copy(abstract_state)
                a_s[current_dim] = divide_point[current_dim][i]
                try:
                    a_s[current_dim + dim] = divide_point[current_dim][i + 1]
                except:
                    print('ssss')
                self.point_to_state(divide_point, current_dim + 1, a_s, state_list)
        else:
            state_list.append(abstract_state)

    def rtree_refinement(self, violate_states, start_id):
        print(self.rtree.bounds)
        all_states = list(self.rtree.intersection(self.rtree.bounds, objects=True))
        print('number of all states in rtree', len(all_states))
        key_dim = len(self.key_dim)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        set_vio = set(violate_states)
        for state in all_states:
            state_str = state.object
            state_str = ','.join([str(_) for _ in state.bbox])
            if state_str in set_vio:
                cnt1 += 1
                if cnt1 % 1000 == 0:
                    print('cnt1', cnt1)
                refine_bounds = bound_bisect(str_to_list(state_str))
                for b in refine_bounds:
                    obj_str = ','.join([str(_) for _ in b[0:key_dim]]) + ',' + ','.join(
                        [str(_) for _ in b[key_dim:2 * key_dim]])
                    # print(obj_str)
                    # rtree.insert(start_id, tuple(b), obj=obj_str)
                    # print(start_id)
                    cnt3 += 1
                    yield start_id, tuple(b), obj_str
                    start_id += 1
            else:
                cnt2 += 1
                yield state.id, tuple(state.bbox), state_str
        print('---------', cnt1, cnt2, cnt3)
        return start_id


if __name__ == "__main__":
    # dt = DivideTool([[-1, 0, 1], [-0.5, 0, 0.5]])
    # abstract_state = list(range(4))
    # state_list = []
    # dt.point_to_state([[-1, 0], [-0.5, 0, 0.5]], 0, abstract_state, state_list)
    # res = dt.intersection([0, -0.3, 0, 0.1])
    # print('eeee')
    dt = initiate_divide_tool_rtree([[-1, 0, -1, -2], [-0.5, 2, -0.5, 2]], [0.1, 0.01, 0.1, 0.2], [1, 2, 3],
                                    'divide_tool_test1')
    # res = dt.intersection([-0.7, 0.2, -0.9, 0, -0.6, 0.3, -0.6, 9])
    res2 = dt.get_abstract_state([-0.6, 1, -0.8, 1])
    vio_states = ['1.01,-0.8,0.2,1.02,-0.7,0.4']

    p = index.Property()
    p.dimension = len(dt.key_dim)
    dt.rtree = index.Index('divide_tool_test1', dt.rtree_refinement(vio_states, dt.rtree.get_size()), properties=p)
    print('size:', dt.rtree.get_size())
    res3 = dt.intersection([-0.65, 1.01, -0.8, 0.2, -0.65, 1.02, -0.7, 0.4])

