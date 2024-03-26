import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import torch


def choose_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return policy_net(state).max(1).indices.view(1, 1).item()


# 创建环境
env = gym.make('CartPole-v1', render_mode="rgb_array")
state, _ = env.reset()
policy_net = torch.jit.load(r"C:\Code\ML\Model\script_policy01.pt")

# 创建一个figure对象
fig = plt.figure()
im = plt.imshow(env.render())

# 初始化帧列表
frames = []
prev_frame = None

# 构建帧列表
for i in range(3000):
    print(i)
    action = choose_action(state)
    state, reward, done, info, _ = env.step(action)

    # 获取新的帧数据
    new_frame = env.render()

    # 检查新帧是否与上一帧相同
    if not np.array_equal(new_frame, prev_frame):
        frames.append(new_frame)
        prev_frame = new_frame

    # # 如果done为True,重置环境
    # if done:
    #     state, _ = env.reset()


# 初始化动画函数
def animate(frame):
    im.set_data(frames[frame])
    print(frame)
    return [im]


# 创建动画对象
ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1, blit=True, repeat=False)

# 显示动画
plt.show()

# 关闭环境
env.close()