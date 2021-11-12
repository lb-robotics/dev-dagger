import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    img = env.render(mode="rgb_array")
    obs, reward, done, infor = env.step(
        env.action_space.sample())  # take a random action
env.close()
