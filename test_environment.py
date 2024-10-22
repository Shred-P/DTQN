import gymnasium as gym
import mobile_env

# 创建环境
env = gym.make("mobile-custom-central-v0", render_mode="human")

# 重置环境并获取初始观测值
obs, info = env.reset()

# 渲染环境
env.render()

# 打印初始观测值
print("初始观测:", obs)

# 进入手动输入动作的循环
done = False
while not done:
    try:
        # 打印动作空间信息
        print(f"可用动作空间: {env.action_space}")

        # 输入动作（根据你的动作空间，输入应该是整数或浮点数）
        action = [int(input("请输入动作: ")),int(input("请输入动作: "))]

        # 在环境中执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        # 渲染当前环境
        env.render()

        # 打印当前的观察值、奖励和其他信息
        print(f"观测值: {obs}, 奖励: {reward}, 结束: {terminated}, 截断: {truncated}")

        # 检查是否终止或截断
        done = terminated or truncated

    except Exception as e:
        print(f"输入无效: {e}")

# 关闭环境
env.close()