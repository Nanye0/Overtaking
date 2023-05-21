import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = gym.make("rl-v0")
env = DummyVecEnv([lambda: env])

# # dqn训练模型定义环境、DQN agent
# model = DQN('MlpPolicy',
#             env,
#             policy_kwargs=dict(net_arch=[256, 256]),
#             learning_rate=5e-4,
#             buffer_size=15000,
#             learning_starts=200,
#             batch_size=64,
#             gamma=0.9,
#             exploration_fraction=0.1,
#             exploration_initial_eps=0.99,
#             exploration_final_eps=0.05,
#             train_freq=1,
#             gradient_steps=1,
#             target_update_interval=50,
#             verbose=1,
#             tensorboard_log="./logs"
#             )
#
# model.learn(total_timesteps=1e5)  # agent的学习
# model.save("highway_dqn_model")  # agent内参数的保存

# del model
# 加载保存在dqn_cartpole.zip中的agent
model = DQN.load("highway_dqn_model", env=env)
# 评估
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# 可视化

# epoch_data = []
# total_reward_data = []

episodes = 1

for eq in range(episodes):
    obs = env.reset()
    done = False
    rewards = 0
    state_date = []
    step = 0
    steps = []
    # print(obs.shape)
    while not done:
        env.render()
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if step <= 28:
            state_date.append(-info[0].get('vehicle_heading'))
            # 航向角 -info[0].get('vehicle_heading') 动作info[0].get('action') 速度info[0].get('speed')
            # 1x abs(obs[0][1][1])*100 2y 5-obs[0][0][2]*10 3vx 4vy obs[0][0][4]
            # print(obs[0][1][1])     10-obs[0][1][1]*100绝对距离

            steps.append(step)
        step += 1
        rewards += reward

    print("Episode : {}, Score : {}".format(eq, rewards))

fig = plt.figure()  # 生成一个画框
ax = fig.add_subplot(1, 1, 1)  # 将画框分为1行1列，并将图 画在画框的第1个位置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.title('航向角变化')  # 侧向距离关系图
plt.xlabel('step')
plt.ylabel('航向角/rad')

# z1 = np.polyfit(steps, state_date, 5)  # 曲线拟合，返回值为多项式的各项系数
# p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
# print(p1)
# y_pred = p1(steps)
# plot1 = ax.plot(steps, state_date, '*', label='original values')
# plot2 = ax.plot(steps, y_pred, 'r', label='fit values')

# steps_smooth = np.linspace(0, 30, 200) # 曲线平滑
# state_date_smooth = make_interp_spline(steps, state_date)(steps_smooth)
# ax.plot(steps_smooth, state_date_smooth, color='r', label='Current')

ax.plot(steps, state_date, color='r', label='Current')
plt.savefig('D:/！！！！毕业设计/基于强化学习的换道超车路径规划研究/pic/test.png')
plt.pause(10)

# fig = plt.figure()  # 生成一个画框
# ax = fig.add_subplot(1, 1, 1)  # 将画框分为1行1列，并将图 画在画框的第1个位置
# plt.title('Reward')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
#
# ax.plot(epoch_data, total_reward_data, color='r', label='Current')
# # ax.plot(epoch_data, total_reward_data)  # 画折线图
# plt.savefig('D:/！！！！毕业设计/reward.png')
# plt.pause(10)
#
