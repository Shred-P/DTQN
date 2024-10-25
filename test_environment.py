import time

import gymnasium as gym
import mobile_env
from utils.logging_utils import RunningAverage, get_logger, custom_get_logger, timestamp
import torch
import numpy as np


def select_action(a, b):
    # 将输入矩阵转换为张量并移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_tensor = torch.tensor(a, device=device)
    b_tensor = torch.tensor(b, device=device)

    # 获取矩阵的形状
    m, n = a_tensor.shape
    actions = []

    for i in range(m):
        # 获取当前行的连接状态和snr值
        connection_status = a_tensor[i]
        snr_values = b_tensor[i]

        # 获取未连接的bs中的snr最大的bs索引
        unconnected_snr_indices = torch.where(connection_status == 0)[0]
        if len(unconnected_snr_indices) > 0:
            # 找到未连接的bs中snr最大的索引
            best_unconnected_idx = unconnected_snr_indices[torch.argmax(snr_values[unconnected_snr_indices])]
            actions.append(best_unconnected_idx.item() + 1)  # 输出1到n表示连接某个bs
        else:
            # 如果所有bs都已连接，输出0表示不操作
            actions.append(0)

    return actions


def get_outout(obs):
    # 将观测值转换为张量并移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_tensor = torch.tensor(obs, device=device)
    obs_tensor = obs_tensor.reshape(5, -1)
    connections = obs_tensor[:, 0:3]
    snrs = obs_tensor[:, 3:6]
    utilities = obs_tensor[:, -1]
    # print(connections)
    action = select_action(connections.cpu().numpy(), snrs.cpu().numpy())

    return action


if __name__ == "__main__":
    wandb_kwargs = {"resume": None}
    # Prepopulate the replay buffer

    arg = {'model': 'human brain',
           'disable_wandb': False, }
    env_str = "mobile-small-central-v0"
    logger = custom_get_logger(project_name='mobile-env', wandb_kwargs=wandb_kwargs)
    seed_config = {"seed": 3407}
    # 创建环境
    env = gym.make(env_str, config=seed_config, render_mode="human")

    # 重置环境并获取初始观测值
    obs, info = env.reset()

    # 渲染环境
    # env.render()

    # 打印初始观测值
    # print("初始观测:", obs)

    # 进入手动输入动作的循环
    done = False
    i = 0
    episode = 3_0000
    eval_f = 50
    # mean_old_reward = RunningAverage(10)
    # mean_new_reward = RunningAverage(10)
    list_reward_o = []
    list_reward_n = []
    for i in range(episode):
        obs, info = env.reset()
        done = False
        total_reward = 0
        new_total_reward = 0
        while not done:
            action = get_outout(obs)
            # 在环境中执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += info.get('old_rewards')
            new_total_reward += reward
            # 渲染当前环境
            # env.render()

            # 打印当前的观察值、奖励和其他信息
            # print(f"观测值: {obs}, 奖励: {reward}, 结束: {terminated}, 截断: {truncated}")

            # 检查是否终止或截断
            done = terminated or truncated
            # time.sleep(0.2)
        list_reward_o.append(total_reward)
        list_reward_n.append(new_total_reward)

        if i % eval_f == 0:
            mean_nr = np.mean(list_reward_n)
            mean_or = np.mean(list_reward_o)
            log_vals = {}
            log_vals.update(
                {
                    f"{env_str}/Return": mean_or,
                    f"{env_str}/new_return": mean_nr,
                }
            )
            list_reward_n = []
            list_reward_o = []
            print(f"for {i} / {episode}:OR_mean = {mean_or}, NR_mean = {mean_nr}")
            logger.log(
                log_vals,
                step=i,
            )

    # 关闭环境
    env.close()
