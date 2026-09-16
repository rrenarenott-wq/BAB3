import gymnasium as gym

# Inisialisasi environment CartPole-v1 dengan render_mode="human"
env = gym.make("CartPole-v1", render_mode="human")

for i_episode in range(20):
    observation, info = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print(f"Episode finished after {t + 1} timesteps")
            break

env.close()