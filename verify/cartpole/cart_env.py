from abstract.cartpole.cartpole_abs import *
import math

from verify.divide_tool import str_to_list


class CartPoleEnv:
    def __init__(self, divide_tool, network):
        self.initial_state = [0.001, 0.001, 0, 0.001]
        self.initial_state_region = None
        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 50
        self.atomic_propositions = ['safe']
        self.formula = 'not(A(G(safe)))'

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.testnum = '0'
        self.divide_tool = divide_tool
        self.network = network

        # 调试信息
        self.rtree_size = 0

    def is_done(self):
        pass

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

    def get_abstract_state_label(self, abstract_state, cnt):
        state_list = str_to_list(abstract_state)
        if state_list[0] < -2.4 or state_list[2] > 2.4 or state_list[1] < -0.21 or state_list[
            3] > 0.21:
            # if cnt <= 1:
            #     print('bad state-----------', state_list)
            return []
        return ['safe']

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

    def next_abstract_domain(self, abstract_obs, act):
        tau = 0.02  # seconds between state updates
        # 获取输入数据
        x0, x1, x_dot0, x_dot1, theta0, theta1, theta_dot0, theta_dot1 = abstract_obs
        # 求f的值
        force = 10.0 if act == 1 else -10.0

        # 确定costheta的上下界，costheta不是单调的，但costheta一定是大于0的
        if theta0 <= 0 and 0 <= theta1:
            costheta1 = 1
            costheta0 = min(math.cos(theta0), math.cos(theta1))
        else:
            costheta0 = min(math.cos(theta0), math.cos(theta1))
            costheta1 = max(math.cos(theta0), math.cos(theta1))

        # 确定sintheta的上下界，sintheta正负未知，但是单调的。
        sintheta0 = math.sin(theta0)
        sintheta1 = math.sin(theta1)
        # 计算角速度的平方的上下界
        if theta_dot0 <= 0 and 0 <= theta_dot1:
            theta_dot_2_0 = 0
            theta_dot_2_1 = max(theta_dot0 ** 2, theta_dot1 ** 2)
        else:
            theta_dot_2_0 = min(theta_dot0 ** 2, theta_dot1 ** 2)
            theta_dot_2_1 = max(theta_dot0 ** 2, theta_dot1 ** 2)

        # 计算temp的上下界
        if 0 <= sintheta0:  # 此时sintheta全部非负
            temp0 = (force + 0.05 * theta_dot_2_0 * sintheta0) / 1.1
            temp1 = (force + 0.05 * theta_dot_2_1 * sintheta1) / 1.1
        elif sintheta1 <= 0:  # 此时sintheta全部非正
            temp0 = (force + 0.05 * theta_dot_2_1 * sintheta0) / 1.1
            temp1 = (force + 0.05 * theta_dot_2_0 * sintheta1) / 1.1
        else:  # 此时sintheta跨越了0
            temp0 = (force + 0.05 * theta_dot_2_1 * sintheta0) / 1.1
            temp1 = (force + 0.05 * theta_dot_2_1 * sintheta1) / 1.1

        # 计算thetaacc分子的上下界
        if 0 <= temp0:  # 此时temp非负
            thetaacc_molecule_0 = 9.8 * sintheta0 - costheta1 * temp1
            thetaacc_molecule_1 = 9.8 * sintheta1 - costheta0 * temp0
        elif temp1 <= 0:  # 此时temp非正
            thetaacc_molecule_0 = 9.8 * sintheta0 - costheta0 * temp1
            thetaacc_molecule_1 = 9.8 * sintheta1 - costheta1 * temp0
        else:  # 此时temp跨越0
            thetaacc_molecule_0 = 9.8 * sintheta0 - costheta1 * temp1
            thetaacc_molecule_1 = 9.8 * sintheta1 - costheta1 * temp0

        # 计算thetaacc的上下界#分母一定是正数
        if 0 <= thetaacc_molecule_0:  # 此时分子非负
            thetaacc0 = thetaacc_molecule_0 / (0.5 * (4.0 / 3.0 - 0.1 * costheta0 ** 2 / 1.1))
            thetaacc1 = thetaacc_molecule_1 / (0.5 * (4.0 / 3.0 - 0.1 * costheta1 ** 2 / 1.1))
        elif thetaacc_molecule_1 <= 0:  # 此时分子非正
            thetaacc0 = thetaacc_molecule_0 / (0.5 * (4.0 / 3.0 - 0.1 * costheta1 ** 2 / 1.1))
            thetaacc1 = thetaacc_molecule_1 / (0.5 * (4.0 / 3.0 - 0.1 * costheta0 ** 2 / 1.1))
        else:  # 此时分子跨越正负：
            thetaacc0 = thetaacc_molecule_0 / (0.5 * (4.0 / 3.0 - 0.1 * costheta1 ** 2 / 1.1))
            thetaacc1 = thetaacc_molecule_1 / (0.5 * (4.0 / 3.0 - 0.1 * costheta1 ** 2 / 1.1))

        if 0 <= thetaacc0:  # 非负
            xacc0 = temp0 - 0.05 * thetaacc1 * costheta1 / 1.1
            xacc1 = temp1 - 0.05 * thetaacc0 * costheta0 / 1.1
        elif thetaacc1 <= 0:  # 非正
            xacc0 = temp0 - 0.05 * thetaacc1 * costheta0 / 1.1
            xacc1 = temp1 - 0.05 * thetaacc0 * costheta1 / 1.1
        else:
            xacc0 = temp0 - 0.05 * thetaacc1 * costheta1 / 1.1
            xacc1 = temp1 - 0.05 * thetaacc0 * costheta1 / 1.1

        # x0, x1, x_dot0, x_dot1, theta0, theta1, theta_dot0, theta_dot1

        new_x0 = x0 + tau * x_dot0
        new_x1 = x1 + tau * x_dot1
        new_x_dot0 = x_dot0 + tau * xacc0
        new_x_dot1 = x_dot1 + tau * xacc1
        new_theta0 = theta0 + tau * theta_dot0
        new_theta1 = theta1 + tau * theta_dot1
        new_theta_dot0 = theta_dot0 + tau * thetaacc0
        new_theta_dot1 = theta_dot1 + tau * thetaacc1

        # 限制越界的情况
        new_x0 = np.clip(new_x0, -4.8, 4.8)
        new_x1 = np.clip(new_x1, -4.8, 4.8)
        new_theta0 = np.clip(new_theta0, -0.418, 0.418)
        new_theta1 = np.clip(new_theta1, -0.418, 0.418)
        new_state = (new_x0, new_x1, new_x_dot0, new_x_dot1, new_theta0, new_theta1, new_theta_dot0, new_theta_dot1)

        return new_state

    def thetaacc_minimum(self, x):
        theta, theta_dot = x[0], x[1]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (self.force_mag + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        return thetaacc

    def thetaacc_maximum(self, x):
        return - self.thetaacc_minimum(x)

    def xacc_minimum(self, x):
        theta, theta_dot = x[0], x[1]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (self.force_mag + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = self.thetaacc_minimum(x)
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return xacc

    def xacc_maximum(self, x):
        return - self.xacc_minimum(x)

    def get_next_states(self, current):
        t0 = time.time()
        cs = current
        try:
            current = str_to_list(current)
        except:
            print(current)
            exit(0)
        if current[0] < -2.4 or current[4] > 2.4 or current[2] < -0.21 or current[6] > 0.21:
            return [cs]
        out = self.network(torch.Tensor(current)).detach()  ##detch()截断反向传播的梯度，[r1,r2]
        action = torch.argmax(out).data.item()  ##[取最大，即取最大值的index]
        p_current = [current[0], current[4], current[1], current[5], current[2], current[6], current[3], current[7]]
        p_next_bounds = self.next_abstract_domain(p_current, action)
        next_bounds = [p_next_bounds[0], p_next_bounds[2], p_next_bounds[4], p_next_bounds[6], p_next_bounds[1],
                       p_next_bounds[3], p_next_bounds[5], p_next_bounds[7]]

        # self.force_mag = 10 if action == 1 else -10
        #
        # bounds = Bounds(current[2: 4], current[6:8])
        # x0 = [(current[2] + current[6]) / 2, (current[3] + current[7]) / 2]
        # xacc_left = minimize(self.xacc_minimum, x0, method='SLSQP', bounds=bounds)
        # xacc_left = self.xacc_minimum(xacc_left.x)
        # xacc_right = minimize(self.xacc_maximum, x0, method='SLSQP', bounds=bounds)
        # xacc_right = - self.xacc_maximum(xacc_right.x)
        #
        # thetaacc_left = minimize(self.thetaacc_minimum, x0, method='SLSQP', bounds=bounds)
        # thetaacc_left = self.thetaacc_minimum(thetaacc_left.x)
        # thetaacc_right = minimize(self.thetaacc_maximum, x0, method='SLSQP', bounds=bounds)
        # thetaacc_right = - self.thetaacc_maximum(thetaacc_right.x)
        #
        # # print('debug', xacc_left, xacc_right)
        # x_left = current[0] + self.tau * current[1]
        # x_right = current[4] + self.tau * current[5]
        # xacc_left = current[1] + self.tau * xacc_left
        # xacc_right = current[5] + self.tau * xacc_right
        # theta_left = current[2] + self.tau * current[3]
        # theta_right = current[6] + self.tau * current[7]
        # thetaacc_left = current[3] + self.tau * thetaacc_left
        # thetaacc_right = current[7] + self.tau * thetaacc_right
        #
        # x_left = np.clip(x_left, -4.8, 4.8)
        # x_right = np.clip(x_right, -4.8, 4.8)
        # theta_left = np.clip(theta_left, -0.42, 0.42)
        # theta_right = np.clip(theta_right, -0.42, 0.42)
        #
        # next_bounds = [x_left, xacc_left, theta_left, thetaacc_left, x_right, xacc_right, theta_right, thetaacc_right]
        next_states = self.divide_tool.intersection(next_bounds)
        # print('bounds计算:', t1 - t0, 'rtree查找抽象状态', t2 - t1, '比例:', ((t1 - t0) / (t2 - t1)))
        return next_states

    # Num     Observation               Min                     Max
    #         0       Cart Position             -4.8                    4.8
    #         1       Cart Velocity             -Inf                    Inf
    #         2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    #         3       Pole Angular Velocity     -Inf                    Inf
    def get_low_dim_state(self, state):
        s = str_to_list(state)
        res = [s[0], s[2], s[4], s[6]]
        obj_str = ','.join([str(_) for _ in res])
        return obj_str
