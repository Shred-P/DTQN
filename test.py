import mobile_env  # 确保你已经安装并导入了 mobile-env
import gymnasium
env = gymnasium.make('mobile-small-central-v0')  # 或者根据 mobile-env 文档中的实际环境名称
# env = gymnasium.make("mobile-small-central-v0", config=config, render_mode="rgb_array")