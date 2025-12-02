import argparse
import torch
import numpy as np
import os
import random
from gymnasium.wrappers import RecordVideo

from config import Config
from dqn import DQNAgent
from utils import to_torch_order, init_env

def evaluate(model_path: str, render: bool = False, num_episodes: int = 30, save_video: bool = False):
    config = Config()
    
    render_mode = "human" if render else "rgb_array"
    
    env = env = init_env(config, render_mode)

    if save_video:
        video_folder = f"videos/{config.env_name}"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    agent = DQNAgent(env, config)
    
    if not os.path.exists(model_path):
        print(f"Error : {model_path} does not exists.")
        return

    print(f"Loading from {model_path}")
    checkpoint = torch.load(model_path, map_location=config.device)
    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.policy_net.eval()

    total_rewards = []
    
    eval_epsilon = 0.05

    print(f"Starting evaluation on {num_episodes} episodes (Epsilon={eval_epsilon})")

    for i in range(num_episodes):
        obs, _ = env.reset()
        state_stack = to_torch_order(obs)
        episode_reward = 0
        done = False
        
        while not done:
            if random.random() < eval_epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(
                        state_stack, dtype=torch.float32, device=config.device
                    ).unsqueeze(0)
                    q_values = agent.policy_net(state_tensor)
                    action = q_values.argmax(1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state_stack = to_torch_order(next_obs)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {i+1}/{num_episodes} : Score = {episode_reward}")

    env.close()

    mean_score = np.mean(total_rewards)
    std_score = np.std(total_rewards)
    max_score = np.max(total_rewards)
    
    print("\n" + "-"*30)
    print("EVALUATION RESULTS")
    print("-"*30)
    print(f"Model : {model_path}")
    print(f"Episodes : {num_episodes}")
    print(f"Mean Score : {mean_score:.2f} +/- {std_score:.2f}")
    print(f"Max Score : {max_score}")
    print("-"*30)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DQN Eval")
    parser.add_argument("--path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render game")
    parser.add_argument("--record", action="store_true", help="Record video")
    
    args = parser.parse_args()

    model_path = args.path
    if model_path is None:
        cfg = Config()
        model_path = f"checkpoints/{cfg.env_name}_dqn_latest.pth"

    evaluate(model_path, render=args.render, num_episodes=args.episodes, save_video=args.record)