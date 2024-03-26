import numpy as np

np.random.seed(42)
point_num = 105462
# 有一个⚪在方形内,我们用 随机点/方形内总点数  估计圆的面积
center = np.array([50, 50])
radius = 20
# 生成随机点
points = np.random.rand(point_num, 2) * 100


def is_in_cycle(point):
    distance = np.linalg.norm(point - center)
    if distance < radius:
        return 1
    else:
        return 0


in_num = 0
for point in points:
    in_num += is_in_cycle(point)

print('prdect: ', (in_num / point_num) * 100**2 )
print('real:  ', np.pi * (radius ** 2))
