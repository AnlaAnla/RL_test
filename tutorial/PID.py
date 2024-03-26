import matplotlib.pyplot as plt
import numpy as np
import time


plt_data1 = []
plt_data2 = []


# 定义PID控制函数
def pid_control(target, current):
    """
    这就是PID控制的主要函数啦!分别计算误差的P,I,D三个部分,最后相加作为输出!
    """
    # 计算误差
    error = target - current

    # 计算P -- 比例项
    p_value = kp * error

    # 计算I -- 积分项
    global integral
    integral = integral + error
    i_value = ki * integral

    # 计算D -- 微分项
    if len(deriv_history) >= 2:
        d_value = kd * (deriv_history[-1] - deriv_history[-2])
    else:
        d_value = 0

    # 计算输出
    u = p_value + i_value + d_value

    # 更新导数历史
    deriv_history.append(error)
    if len(deriv_history) > 10:
        deriv_history.pop(0)

    return u

def show():
    length = len(plt_data2)
    x = [x for x in range(length)]
    plt_data1 = [target] * length

    plt.plot(x, plt_data1)
    plt.plot(x, plt_data2)
    plt.show()

if __name__ == '__main__':
    # 目标值,就是我们希望系统达到的状态值
    target = 100

    # 当前值,就是系统当前的实际状态值
    current = 0

    # PID参数
    kp = 0.5  # 比例常数P(Proportion)
    ki = 0.1  # 积分常数I(Integration)
    kd = 0.2  # 微分常数D(Differentiation)

    # 初始化
    integral = 0
    deriv_history = []

    # 开始控制循环!
    start_time = time.time()
    while True:
        # 计算控制量
        output = pid_control(target, current)

        # 更新当前值
        current += output

        plt_data2.append(current)

        # 打印当前值(通过在终端中打印一些可爱的符号来模拟实时值)
        print(f"\r💗💗💗Current: {'🌸' * int(current / 10)}💗💗💗", end="")

        # 判断是否到达目标
        if abs(current - target) < 1:
            print(f"\n \n\n恭喜到达目标值啦!用时 {time.time() - start_time:.2f} 秒!🎉🎉🎉")
            break

        time.sleep(0.1)  # 暂停0.1s,防止输出太快

    show()