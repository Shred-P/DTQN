import gymnasium as gym
import mobile_env
from pynput import keyboard
import matplotlib.pyplot as plt

# 创建环境，使用 "rgb_array" 渲染模式
env = gym.make("mobile-custom-central-v0", render_mode="human")

# 重置环境并获取初始观测值
obs, info = env.reset()

# 打印初始观测值
print("初始观测:", obs)

done = False
env.render()
# 设置渲染窗口
# plt.ion()
# fig, ax = plt.subplots()
# im = ax.imshow(env.render())

# 定义动作映射
action_mapping = {
    'q': [0],
    'w': [1],
    'e': [2],
    'r': [3],
}

def on_press(key):
    global done
    if done:
        return False

    try:
        # 获取按键字符
        if hasattr(key, 'char') and key.char in action_mapping:
            action = action_mapping[key.char]

            # 在环境中执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # 更新渲染
            # im.set_data(env.render())
            # plt.pause(0.001)

            # 打印当前的观察值、奖励和其他信息
            print(f"观测值: {obs}, 奖励: {reward}, 结束: {terminated}, 截断: {truncated}")

            # 检查是否终止或截断
            done = terminated or truncated
            if done:
                return False

    except Exception as e:
        print(f"错误: {e}")

# 启动键盘监听
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# 关闭环境
env.close()
# plt.ioff()
# plt.show()
