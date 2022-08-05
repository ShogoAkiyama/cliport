import numpy as np
import random

from cliport.dataset.dataset import RavensDataset
from cliport.envs.envs import Environment


def main():
    env = Environment()

    # Initialize scripted oracle agent and dataset.
    agent = env.task.oracle()
    data_path = "./data/packing-boxes-pairs-seen-colors-train"
    dataset = RavensDataset(data_path, n_demos=0)

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed

    if seed < 0:
        seed = -2

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < 1000:
        episode = []
        total_reward = 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        # env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Rollout expert policy
        for _ in range(env.task.max_steps):
            act = agent.act()
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward

            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')

            if done:
                break

        episode.append((obs, None, reward, info))

        # # Only save completed demonstrations.
        # if total_reward > 0.99:
        #     dataset.add(seed, episode)


if __name__ == '__main__':
    main()
