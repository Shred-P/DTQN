import numpy as np


def total_dimensions(num_actions_per_dim):
    """
    计算多维动作空间编码成整数后的总维度。

    参数:
    - num_actions_per_dim: 每个维度的动作数量，比如 [4, 4, 4, 4, 4]

    返回:
    - 编码成整数后的总维度
    """
    return np.prod(num_actions_per_dim)


# 示例用法
num_actions_per_dim = [4, 4, 4, 4, 4]  # 每个维度的动作空间大小
total_dim = total_dimensions(num_actions_per_dim)
print(f"Total dimensions: {total_dim}")
