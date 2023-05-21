# 设计旁车固定的换道超车
import highway_env
# from highway_env.envs.common.abstract import AbstractEnv
# from highway_env.road.road import Road, RoadNetwork
# from highway_env.envs.common.action import Action
# from highway_env.vehicle.controller import ControlledVehicle
# from highway_env.vehicle.kinematics import Vehicle
#
#
# LANES = 2
# ANGLE = 0
# START = 0
# LENGHT = 200
# SPEED_LIMIT = 30
# SPEED_REWARD_RANGE = [10, 30]
# COL_REWARD = -1
# HIGH_SPEED_REWARD = 0
# RIGHT_LANE_REWARD = 0
# DURATION = 100
#
#
# class myEnv(AbstractEnv):
#
#     @classmethod
#     def default_config(cls) -> dict:
#         config = super().default_config()
#         config.update(
#             {
#                 'observation': {
#                     'type': 'Kinematics',
#                     "absolute": False,
#                     "normalize": False
#                 },
#                 'action': {'type': 'DiscreteMetaAction'},
#
#                 "reward_speed_range": SPEED_REWARD_RANGE,
#                 "simulation_frequency": 20,
#                 "policy_frequency": 20,
#                 "centering_position": [0.3, 0.5],
#             }
#         )
#         return config
#
#     def _reset(self) -> None:
#         self._create_road()
#         self._create_vehicles()
#
#
#     def _create_road(self) -> None:
#         self.road = Road(
#             network=RoadNetwork.straight_road_network(LANES, speed_limit=SPEED_LIMIT),
#             np_random=self.np_random,
#             record_history=False,
#         )
#
# 	# 创建车辆
#     def _create_vehicles(self) -> None:
#
#         vehicle = Vehicle.create_random(self.road, speed=23, lane_id=1, spacing=0.3)
#         vehicle = self.action_type.vehicle_class(
#             self.road,
#             vehicle.position,
#             vehicle.heading,
#             vehicle.speed,
#         )
#         self.vehicle = vehicle
#         self.road.vehicles.append(vehicle)
#
#         vehicle = Vehicle.create_random(self.road, speed=30, lane_id=1, spacing=0.35)
#         vehicle = self.action_type.vehicle_class(
#             self.road,
#             vehicle.position,
#             vehicle.heading,
#             vehicle.speed,
#         )
#         self.road.vehicles.append(vehicle)
#
# 	# 重写的奖励函数，仅考虑车辆碰撞影响
#     def _reward(self, action: Action) -> float:
#         reward = 0
#
#         lane = (
#             self.vehicle.target_lane_index[2]
#             if isinstance(self.vehicle, ControlledVehicle)
#             else self.vehicle.lane_index[2]
#         )
#
#         if self.vehicle.crashed:
#             reward = -1
#         elif lane == 0:
#             reward += 1
#
#         reward = 0 if not self.vehicle.on_road else reward
#
#         return reward
#
#     def _is_terminal(self) -> bool:
#         return (
#             self.vehicle.crashed
#             or self.time >= DURATION
#             or (False and not self.vehicle.on_road)
#         )
#
#
# if __name__ == '__main__':
#     env = myEnv()
#     obs = env.reset()
#
#     eposides = 100
#     rewards = 0
#     for eq in range(eposides):
#         obs = env.reset()
#         env.render()
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             obs, reward, done, info = env.step(action)
#             env.render()
#             rewards = reward
#         print(rewards)


# 无设置旁车的highway-v0
# import gym
# import highway_env
# import numpy as np
#
# env = gym.make("rl-v0")
# env.configure({
#
# })
# state = env.reset()  # 状态初始化
# state_space = env.observation_space.shape[0]  # 状态空间的大小
#
# act_space = env.action_space.n  # 动作空间的大小
#
# # Q-learning
# Q = np.zeros([state_space, act_space])  # 创建一个Q-table
# # Q = np.random.uniform(low=0, high=1, size=(state_space ** 5, act_space))
# NUM_DIGITIZED = 6
# # GAMMA就是Q学习中的折扣率（0~1），用以表示智能体对长期回报的看法。GAMMA为0，表示只看当前的回报。GAMMA为1，则是极其重视长期回报。
# GAMMA = 0.99
# # 学习率。ETA越大，则进行每一步更新时，受reward影响更多。
# ETA = 0.5
# # 假如连续控制200次，游戏还没结束，视为成功通关。
# MAX_STEPS = 200
# # 总共进行2000次训练。（不一定会训练2000次，详情见env.run，有详解。）
# NUM_EPISODES = 2000
#
#
# def digitize_state(self, observation):
#     cart_pos, cart_v, pole_angle, pole_v = observation
#     # 离散化变量依靠np.digitize。
#     # 比如：np.digitize(-1.5,[-1.6, -0.8,  0. ,  0.8,  1.6])=1
#     #     np.digitize(-2.7,[-1.6, -0.8,  0. ,  0.8,  1.6])=0
#     #     np.digitize(-0.77,[-1.6, -0.8,  0. ,  0.8,  1.6])=2
#     # 也就是落在这个区间里。虽然边界值区间是（-2.4，-1.6），但是我们认为-∞到-1.6为一个区间。视为游戏结束。
#     digitized = [
#         np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIGITIZED)),
#         np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZED)),
#         np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIGITIZED)),
#         np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZED))
#     ]
#     # 每一个状态由0~5表示。每一刻的状态由[2,1,2,4]这样的列表表示。
#     # 这里是用1296个数表示状态。（6进制）  其位权就是其索引，就是6的i次方。
#     # 可以理解为对1296个状态进行编码
#     return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitized)])
#
#
# def update_Q_table(self, observation, action, reward, observation_next):
#     # 用observation获取编码后的状态，也就是利用前面写好的digitize函数。
#     state = self.digitize_state(observation)
#     # 获取下一次状态（依然是编码之后）。
#     state_next = self.digitize_state(observation_next)
#     # 如之前所述，q_table是一个1296*2的矩阵。用state_next索引到那一行的两个动作。求出两个动作中Q值最大的那个动作。（动作价值）
#     # 这也是区别去SARSA算法的地方。
#     Max_Q_next = max(self.q_table[state_next][:])
#     # 更新Q表格（这里出现了之前定义的ETA和GAMMA，也就是学习率和折扣率，详情见上面）
#     self.q_table[state, action] = self.q_table[state, action] + \
#                                   ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
#     # 动作决策，输入参数是当前状态，以及是第几轮训练。
#
#
# def decide_action(self, observation, episode):
#     # 依然是digitize_state函数。（所有拿到的observation观测值都无法直接使用，必须用digitiaze_state转换为可以使用的state）
#     state = self.digitize_state(observation)
#     # 这里实现的是随机选取动作，你可能会疑问，为啥随机选取动作的概率还得随训练轮数变换？
#     # 其意义在于前期让智能体更大胆的尝试，后期学到比较好的Q表格后再应用知识。
#     # 这样更有利于泛化。
#     epsilon = 0.5 * (1 / (episode + 1))
#     if epsilon <= np.random.uniform(0, 1):
#         action = np.argmax(self.q_table[state][:])
#     else:
#         action = np.random.choice(self.num_actions)
#     return action
#
#     # 返回值是4，代表着Cart的位置和速度，Pole的角度和角速度四种状态变量。
#     num_states = env.observation_space.shape[0]  # 4
#     # 返回值是2，代表着运动。分别为向左和向右。
#     num_actions = env.action_space.n
#
#
# def run(self):
#     # 一个中间变量，存放已经连续通关的局数。当连续通关10次，结束训练（也有可能直到2000次训练，依然没出现连赢10把。不过连续玩2000局也会自动结束。）。
#     complete_episodes = 0
#     # 训练是否结束。
#     is_episode_final = False
#     # 空
#     frames = []
#     # 开始进行训练，上文中这里设置的是2000（NUM_EPISODES），也就是2000次。
#     for episode in range(NUM_EPISODES):
#         # 环境重置。要使用gym，必须先调用一次。无论是之前游戏刚结束 还是第一次调用环境。同时会返回一个observation（或者说state）
#         observation = self.env.reset()  # initialize environment
#         # 每一轮控制200（MAX_STEPS）步，结束游戏。
#         for step in range(MAX_STEPS):
#
#             # 智能体观测环境，做出决策
#             action = self.agent.get_action(observation, episode)
#             # 返回值有四个，分别状态、奖励值、游戏状态（False游戏进行，True表示游戏截至）、信息（CartPole这个环境没啥附加信息，别的会有）
#             # 环境给出反馈reward，并更新状态。当然，这次强化学习任务并不需要这里的奖励和信息俩个量。我们另设置了奖励，为一轮游戏结束时结算奖励。
#             observation_next, _, done, _ = self.env.step(action)  # reward and info not need
#             # 游戏结束，则进行结算。
#             if done:
#                 # 小于180步结束游戏时
#                 if step < 180:
#                     # 奖励值为负
#                     reward = -1
#                     # 连续胜场清楚为0
#                     complete_episodes = 0
#                     # 超过180步时，注意，这里有时是180步-200步之间停止游戏，也就是未通关，但依然计算reward。这样做是为了加快收敛。
#                 # 可以通关调整这个参数，观察结果。增加180到200，会导致收敛缓慢。
#                 else:
#                     # 奖励值为正数。
#                     reward = 1
#                     # 连续胜场加1。
#                     complete_episodes += 1
#                     # 游戏依然进行
#             else:
#                 # 奖励值为0
#                 reward = 0
#                 # 更新Q表格（所需要的参数为上一次状态，上一次执行的策略，当前状态，奖励值，）
#             self.agent.update_Q_function(observation, action, reward, observation_next)
#             # 更新状态变量
#             observation = observation_next
#             # 如果游戏结束（本轮游戏）
#             if done:
#                 # 输出这是第几轮训练，进行了多少步
#                 print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
#                 # 蹦出循环
#                 break
#         # 整个训练结束（条件为连续10轮超过200步，或者进行了2000轮训练。）
#         if is_episode_final is True:
#             # 跳出循环
#             break
#         # 当连胜超过10轮， is_episode_final调整为True
#         if complete_episodes >= 10:
#             print('succeeded for 10 times')
#             is_episode_final = True
#
#
# env.run()

# for episode in range(1, 2000):
#     done = False
#     reward = 0  # 瞬时reward
#     R_cum = 0  # 累计reward
#     state = env.reset()  # 状态初始化
#     while done != True:
#         action = np.argmax(Q[state].astype('int32'))
#         state2, reward, done, info = env.step(action)
#         Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
#         R_cum += reward
#         state = state2
#         # env.render()
#     if episode % 50 == 0:
#         print('episode:{};total reward:{}'.format(episode, R_cum))
#
# print('The Q table is:{}'.format(Q))
#
# # 测试阶段
# conter = 0
# reward = None
# while conter < 200:
#     obs = env.reset()
#     env.render()
#     done = False
#     while not done:
#         # action = np.argmax(Q[state])
#         action = np.argmax(Q[state].astype('int32'))
#         obs, reward, done, info = env.step(action)
#         conter = conter + 1
#     print(reward)

# import gym
# import highway_env
# import numpy as np
#
# env = gym.make("rl-v0")
#
# NUM_DIGITIZED = 6
# # GAMMA就是Q学习中的折扣率（0~1），用以表示智能体对长期回报的看法。GAMMA为0，表示只看当前的回报。GAMMA为1，则是极其重视长期回报。
# GAMMA = 0.99
# # 学习率。ETA越大，则进行每一步更新时，受reward影响更多。
# ETA = 0.5
# # 假如连续控制200次，游戏还没结束，视为成功通关。
# MAX_STEPS = 200
# # 总共进行2000次训练。（不一定会训练2000次，详情见env.run，有详解。）
# NUM_EPISODES = 2000
#
#
# # 定义Brain类，智能体与环境交互的主要实现途径。
# class Brain:
#     # num_states为4，代表着Cart的位置和速度，Pole的角度和角速度四种状态变量。num_action为2，分别为向左和向右。
#     # 这俩个参数从环境env中获取，然后传入Brain.
#     def __init__(self, num_states, num_actions):
#         self.num_actions = num_actions
#         # 创建一个Q表格，也就是我们的Q函数。这里是一个(6^5,2)格式的矩阵。
#         # 其中所有数字为0~1的随机数。这是一种启动方式，也可以全部为0。毕竟只是初始化一个表格，最终都能通过学习收敛。当然，如果通过设置初始化值
#         # 加快收敛速度，也是一个研究方向。
#         self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED ** num_states, num_actions))
#         # 离散化。因为我们的状态是连续的，比如Cart的速度。但表格里只能装下有限的状态，所以通过bins创建一个列表，表示区间。
#
#     def bins(self, clip_min, clip_max, num):
#         # 比如小车的极限位置是-2.4到2.4，np.linspace(-2.4, 2.4, 7)[1: -1] =array([-1.6, -0.8,  0. ,  0.8,  1.6])
#         return np.linspace(clip_min, clip_max, num + 1)[1: -1]
#         # 用四个变量（Cart位置、速度Pole位置、速度）提取状态observation
#
#     def digitize_state(self, observation):
#         cart_pos, cart_v, pole_angle, pole_v = observation
#         # 离散化变量依靠np.digitize。
#         # 比如：np.digitize(-1.5,[-1.6, -0.8,  0. ,  0.8,  1.6])=1
#         #     np.digitize(-2.7,[-1.6, -0.8,  0. ,  0.8,  1.6])=0
#         #     np.digitize(-0.77,[-1.6, -0.8,  0. ,  0.8,  1.6])=2
#         # 也就是落在这个区间里。虽然边界值区间是（-2.4，-1.6），但是我们认为-∞到-1.6为一个区间。视为游戏结束。
#         digitized = [
#             np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIGITIZED)),
#             np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZED)),
#             np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIGITIZED)),
#             np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZED))
#         ]
#         # 每一个状态由0~5表示。每一刻的状态由[2,1,2,4]这样的列表表示。
#         # 这里是用1296个数表示状态。（6进制）  其位权就是其索引，就是6的i次方。
#         # 可以理解为对1296个状态进行编码
#         return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitized)])
#
#     # 更新Q表格，也就是根据state和reward
#     def update_Q_table(self, observation, action, reward, observation_next):
#         # 用observation获取编码后的状态，也就是利用前面写好的digitize函数。
#         state = self.digitize_state(observation)
#         # 获取下一次状态（依然是编码之后）。
#         state_next = self.digitize_state(observation_next)
#         # 如之前所述，q_table是一个1296*2的矩阵。用state_next索引到那一行的两个动作。求出两个动作中Q值最大的那个动作。（动作价值）
#         # 这也是区别去SARSA算法的地方。
#         Max_Q_next = max(self.q_table[state_next][:])
#         # 更新Q表格（这里出现了之前定义的ETA和GAMMA，也就是学习率和折扣率，详情见上面）
#         self.q_table[state, action] = self.q_table[state, action] + \
#                                       ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
#         # 动作决策，输入参数是当前状态，以及是第几轮训练。
#
#     def decide_action(self, observation, episode):
#         # 依然是digitize_state函数。（所有拿到的observation观测值都无法直接使用，必须用digitiaze_state转换为可以使用的state）
#         state = self.digitize_state(observation)
#         # 这里实现的是随机选取动作，你可能会疑问，为啥随机选取动作的概率还得随训练轮数变换？
#         # 其意义在于前期让智能体更大胆的尝试，后期学到比较好的Q表格后再应用知识。
#         # 这样更有利于泛化。
#         epsilon = 0.5 * (1 / (episode + 1))
#         if epsilon <= np.random.uniform(0, 1):
#             action = np.argmax(self.q_table[state][:])
#         else:
#             action = np.random.choice(self.num_actions)
#         return action
#


# eposides = 1000
# rewards = 0
# for eq in range(eposides):
#     rewards = 0
#     obs = env.reset()
#     env.render()
#     done = False
#     while not done:
#         # action = env.action_space.sample()
#         action = np.argmax(Q[state])
#         obs, reward, done, info = env.step(action)
#         env.render()
#         rewards += reward
#     print(rewards)


# 测试highway-v0的动作状态设置参数
# import gym
# import highway_env
# import pprint
#
# env = gym.make('highway-v0')
# env.reset()
# pprint.pprint(env.config)


# import gym
# import numpy as np
# import highway_env
# from matplotlib import pyplot as plt
#
# env = gym.make("rl-v0")
#
# # Q-Learning settings
# LEARNING_RATE = 0.1
# DISCOUNT = 0.95
# EPISODES = 25000
#
# SHOW_EVERY = 1000
#
# # Exploration settings
# epsilon = 1  # not a constant, qoing to be decayed
# START_EPSILON_DECAYING = 1
# END_EPSILON_DECAYING = EPISODES//2
# epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
#
# DISCRETE_OS_SIZE = [20, 20]
# discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
#
# def get_discrete_state(state):
#     discrete_state = (state - env.observation_space.low)/discrete_os_win_size
#     return tuple(discrete_state.astype(np.int64))  # we use this tuple to look up the 3 Q values for the available actions in the q-table
#
# q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#
# for episode in range(EPISODES):
#     state = env.reset()
#     discrete_state = get_discrete_state(state)
#
#     if episode % SHOW_EVERY == 0:
#         render = True
#         print(episode)
#     else:
#         render = False
#
#     done = False
#     while not done:
#         if np.random.random() > epsilon:
#             # Get action from Q table
#             action = np.argmax(q_table[discrete_state])
#         else:
#             # Get random action
#             action = np.random.randint(0, env.action_space.n)
#
#         new_state, reward, done, _ = env.step(action)
#         new_discrete_state = get_discrete_state(new_state)
#
#         # If simulation did not end yet after last step - update Q table
#         if not done:
#
#             # Maximum possible Q value in next step (for new state)
#             max_future_q = np.max(q_table[new_discrete_state])
#
#             # Current Q value (for current state and performed action)
#             current_q = q_table[discrete_state + (action,)]
#
#             # And here's our equation for a new Q value for current state and action
#             new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#
#             # Update Q table with new Q value
#             q_table[discrete_state + (action,)] = new_q
#
#         # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
#         elif new_state[0] >= env.goal_position:
#             q_table[discrete_state + (action,)] = 0
#             print("we made it on episode {}".format(episode))
#
#         discrete_state = new_discrete_state
#
#         if render:
#             env.render()
#
#     # Decaying is being done every episode if episode number is within decaying range
#     if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
#         epsilon -= epsilon_decay_value
#
# env.close()
# # np.save(path, q_table) # path自己指定


import gym
import highway_env
import numpy as np


NUM_DIGITIZED = 6
# GAMMA就是Q学习中的折扣率（0~1），用以表示智能体对长期回报的看法。GAMMA为0，表示只看当前的回报。GAMMA为1，则是极其重视长期回报。
GAMMA = 0.99
# 学习率。ETA越大，则进行每一步更新时，受reward影响更多。
ETA = 0.5
# 假如连续控制200次，游戏还没结束，视为成功通关。?
MAX_STEPS = 200
# 总共进行2000次训练。（不一定会训练2000次，详情见env.run，有详解。）
NUM_EPISODES = 2000
num_episodes = 1000
epsilon = 0.1

# state = env.reset()# 测试状态类型
# p, x, y, vx, vy = state[0]
# print(p, x, y, vx, vy )
# print(state[1])


# def discretize(observation):
#     # 将车辆位置离散化为5个区间
#     # p = observation[0][0]
#
#     x_bins = np.arange(-1, 1, 0.1)  # 20
#     y_bins = np.arange(0, 0.5, 0.1)  # 5
#     # 将车辆速度离散化为3个区间
#     vx_bins = np.arange(-0.2, 0.4, 0.03)   # 20
#     vy_bins = np.arange(-0.03, 0.03, 0.012)  # 5
#
#     x = np.digitize(observation[0][1], x_bins)
#     y = np.digitize(observation[0][2], y_bins)
#
#     vx = np.digitize(observation[1][3], vx_bins)
#     vy = np.digitize(observation[1][4], vy_bins)
#     return [x, y, vx, vy]


def digitize_state(self, state):#
    p, x, y, vx, vy = state[0]
    print(p, x, y, vx, vy)
    # 离散化变量依靠np.digitize。
    # 比如：np.digitize(-1.5,[-1.6, -0.8,  0. ,  0.8,  1.6])=1
    #     np.digitize(-2.7,[-1.6, -0.8,  0. ,  0.8,  1.6])=0
    #     np.digitize(-0.77,[-1.6, -0.8,  0. ,  0.8,  1.6])=2
    # 也就是落在这个区间里。虽然边界值区间是（-2.4，-1.6），但是我们认为-∞到-1.6为一个区间。视为游戏结束。
    digitized = [
        np.digitize(x, bins=self.bins(-1, 1, NUM_DIGITIZED)),
        np.digitize(y, bins=self.bins(0, 0.5, NUM_DIGITIZED)),
        np.digitize(vx, bins=self.bins(0.1, 0.4, NUM_DIGITIZED)),
        np.digitize(vy, bins=self.bins(-0.03, 0.03, NUM_DIGITIZED))
    ]
    # 每一个状态由0~5表示。每一刻的状态由[2,1,2,4]这样的列表表示。
    # 这里是用1296个数表示状态。（6进制）  其位权就是其索引，就是6的i次方。
    # 可以理解为对1296个状态进行编码
    return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitized)])


# def update_Q_table(self, observation, action, reward, observation_next):
#     # 用observation获取编码后的状态，也就是利用前面写好的digitize函数。
#     state = self.digitize_state(observation)
#     # 获取下一次状态（依然是编码之后）。
#     state_next = self.digitize_state(observation_next)
#     # 如之前所述，q_table是一个1296*2的矩阵。用state_next索引到那一行的两个动作。求出两个动作中Q值最大的那个动作。（动作价值）
#     # 这也是区别去SARSA算法的地方。
#     Max_Q_next = max(self.q_table[state_next][:])
#     # 更新Q表格（这里出现了之前定义的ETA和GAMMA，也就是学习率和折扣率，详情见上面）
#     self.q_table[state, action] = self.q_table[state, action] + \
#                                   ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
#     # 动作决策，输入参数是当前状态，以及是第几轮训练。


# def decide_action(self, observation, episode):
#     # 依然是digitize_state函数。（所有拿到的observation观测值都无法直接使用，必须用digitiaze_state转换为可以使用的state）
#     state = self.digitize_state(observation)
#     # 这里实现的是随机选取动作，你可能会疑问，为啥随机选取动作的概率还得随训练轮数变换？
#     # 其意义在于前期让智能体更大胆的尝试，后期学到比较好的Q表格后再应用知识。
#     # 这样更有利于泛化。
#     epsilon = 0.5 * (1 / (episode + 1))
#     if epsilon <= np.random.uniform(0, 1):
#         action = np.argmax(self.q_table[state][:])
#     else:
#         action = np.random.choice(self.num_actions)
#     return action


env = gym.make("rl-v0")
# observation_space = env.observation_space.shape[0]
# action_space = env.action_space.n
# print(observation_space)
#
# print(env.action_space.n)
# Q-table initialization
# q_table = np.zeros((observation_space, action_space))


# 返回值是4，代表着状态空间。
num_states = 4  # 4
# 返回值是5，代表着动作空间。
num_actions = env.action_space.n

q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED ** num_states, num_actions))


for episode in range(num_episodes):
    state = env.reset()
    discrete_state = digitize_state(state)
    print(discrete_state)
    done = False
    env.render()
    while not done:

        # Choose action based on epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(q_table[discrete_state])

        # Perform action and get next state and reward
        next_state, reward, done, info = env.step(action)

        # Update Q-table using Q-learning formula
        q_table[discrete_state][action] = (1 - ETA) * q_table[discrete_state][action] + ETA * (
                reward + GAMMA * np.max(q_table[digitize_state(next_state)]))

        state = digitize_state(next_state)

print("Training completed.")

# Evaluate trained agent
total_reward = 0
state = env.reset()
done = False

while not done:
    action = np.argmax(q_table[discrete_state])
    state, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")




# def run(self):
#     # 一个中间变量，存放已经连续通关的局数。当连续通关10次，结束训练（也有可能直到2000次训练，依然没出现连赢10把。不过连续玩2000局也会自动结束。）。
#     complete_episodes = 0
#     # 训练是否结束。
#     is_episode_final = False
#     # 空
#     frames = []
#     # 开始进行训练，上文中这里设置的是2000（NUM_EPISODES），也就是2000次。
#     for episode in range(NUM_EPISODES):
#         # 环境重置。要使用gym，必须先调用一次。无论是之前游戏刚结束 还是第一次调用环境。同时会返回一个observation（或者说state）
#         observation = self.env.reset()  # initialize environment
#         # 每一轮控制200（MAX_STEPS）步，结束游戏。
#         for step in range(MAX_STEPS):
#
#             # 智能体观测环境，做出决策
#             action = self.agent.get_action(observation, episode)
#             # 返回值有四个，分别状态、奖励值、游戏状态（False游戏进行，True表示游戏截至）、信息（CartPole这个环境没啥附加信息，别的会有）
#             # 环境给出反馈reward，并更新状态。当然，这次强化学习任务并不需要这里的奖励和信息俩个量。我们另设置了奖励，为一轮游戏结束时结算奖励。
#             observation_next, reward, done, info = self.env.step(action)  # reward and info not need
#
#             # 游戏结束，则进行结算。
#             # if done:
#             #     # 小于180步结束游戏时
#             #     if step < 180:
#             #         # 奖励值为负
#             #         reward = -1
#             #         # 连续胜场清楚为0
#             #         complete_episodes = 0
#             #         # 超过180步时，注意，这里有时是180步-200步之间停止游戏，也就是未通关，但依然计算reward。这样做是为了加快收敛。
#             #     # 可以通关调整这个参数，观察结果。增加180到200，会导致收敛缓慢。
#             #     else:
#             #         # 奖励值为正数。
#             #         reward = 1
#             #         # 连续胜场加1。
#             #         complete_episodes += 1
#             #         # 游戏依然进行
#             # else:
#             #     # 奖励值为0
#             #     reward = 0
#             #     # 更新Q表格（所需要的参数为上一次状态，上一次执行的策略，当前状态，奖励值，）
#             self.agent.update_Q_function(observation, action, reward, observation_next)
#             # 更新状态变量
#             observation = observation_next
#             # 如果游戏结束（本轮游戏）
#             if done:
#                 # 输出这是第几轮训练，进行了多少步
#                 print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
#                 # 蹦出循环
#                 break
#         # 整个训练结束（条件为连续10轮超过200步，或者进行了2000轮训练。）
#         if is_episode_final is True:
#             # 跳出循环
#             break
#         # 当连胜超过10轮， is_episode_final调整为True
#         if complete_episodes >= 10:
#             print('succeeded for 10 times')
#             is_episode_final = True
#
#
# env.run()
#
# for episode in range(1, 2000):
#     done = False
#     reward = 0  # 瞬时reward
#     R_cum = 0  # 累计reward
#     state = env.reset()  # 状态初始化
#     while done != True:
#         action = np.argmax(Q[state].astype('int32'))
#         state2, reward, done, info = env.step(action)
#         Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
#         R_cum += reward
#         state = state2
#         # env.render()
#     if episode % 50 == 0:
#         print('episode:{};total reward:{}'.format(episode, R_cum))
#
# print('The Q table is:{}'.format(Q))
#
# # 测试阶段
# conter = 0
# reward = None
# while conter < 200:
#     obs = env.reset()
#     env.render()
#     done = False
#     while not done:
#         # action = np.argmax(Q[state])
#         action = np.argmax(Q[state].astype('int32'))
#         obs, reward, done, info = env.step(action)
#         conter = conter + 1
#     print(reward)
#
#
#
#
#
#
#
# import gym
# import numpy as np
# from highway_env.utils import near_split
#
#
# # 定义离散化方法
# def discretize(observation):
#     # 将车辆位置离散化为5个区间
#     x_bins = np.arange(-1, 1, 0.1)
#     y_bins = np.arange(0, 0.5, 0.1)
#     # 将车辆速度离散化为3个区间
#     vx_bins = np.arange(-0.2, 0.4, 0.03)
#     vy_bins = np.arange(-0.03, 0.03, 0.012)
#
#     x = np.digitize(observation[0][1], x_bins)
#     y = np.digitize(observation[0][2], y_bins)
#
#     vx = np.digitize(observation[1][3], vx_bins)
#     vy = np.digitize(observation[1][4], vy_bins)
#     return [x, y, vx, vy]
#
#
# # 创建Highway-env环境
# env = gym.make("rl-v0")
#
# # 获取初始状态
# observation = env.reset()
#
# # 对状态空间进行离散化
# discrete_observation = discretize(observation)
#
# # 打印离散化后的状态空间
# print("离散化前的状态空间：", observation)
# print("离散化后的状态空间：", discrete_observation)


# env.configure({
#     'observation': {
#         'type': 'Kinematics',
#         "absolute": False,
#         "normalize": False
#     },
#     'action': {'type': 'DiscreteMetaAction'},
#     'simulation_frequency': 15,
#     'policy_frequency': 10,
#     'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
#     'screen_width': 600,
#     'screen_height': 150,
#     'centering_position': [0.3, 0.5],
#     'scaling': 5.5,
#     'show_trajectories': False,
#     'render_agent': True,
#     'offscreen_rendering': False,
#     'manual_control': False,
#     'real_time_rendering': False,
#     'lanes_count': 2,  # 车道数
#     'vehicles_count': 1,  # 周围车辆数
#     'controlled_vehicles': 1,
#     'initial_lane_id': None,
#     'duration': 100,
#     'ego_spacing': 2,
#     'vehicles_density': 1,
#     'collision_reward': -1,
#     'right_lane_reward': 0.1,
#     'high_speed_reward': 0.4,
#     'lane_change_reward': 0,
#     'reward_speed_range': [20, 30],
#     'offroad_terminal': False,
#     'lane_from': 1
# })
