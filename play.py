"""
play.py

Load a trained DQN model and watch it play using the greedy policy (deterministic=True).

Examples
--------
# Watch the CNN agent play
python play.py --policy cnn --model models/baseline/dqn_model.zip

# Watch the CNN agent and save an MP4 video
python play.py --policy cnn --model models/baseline/dqn_model.zip --record

# Watch the MLP (RAM) agent play
python play.py --policy mlp --model models/mlp/dqn_model.zip
"""

import argparse
import moviepy
import os
import time
import ale_py
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

PIXEL_ENV = "ALE/Pong-v5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN Atari agent.")
    parser.add_argument(
        "--policy", choices=["cnn", "mlp"], default="cnn", help="Policy used during training.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model (.zip).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play.")
    parser.add_argument("--fps", type=int, default=60, help="Playback speed.")
    parser.add_argument("--record", action="store_true",
                        help="Record gameplay to an MP4 (CNN only).")
    return parser.parse_args()


def build_render_env(policy, record=False):
    render_mode = "rgb_array" if record else "human"

    if policy == "cnn":
        env = make_atari_env(PIXEL_ENV, n_envs=1, seed=42,
                             env_kwargs={"render_mode": render_mode})
        env = VecFrameStack(env, n_stack=4)

        if record:
            os.makedirs("videos", exist_ok=True)
            env = VecVideoRecorder(
                env,
                video_folder="videos",
                record_video_trigger=lambda step: step == 0,
                video_length=5000,
                name_prefix="dqn-play",
            )
    else:
        if record:
            print(
                "\nWarning: Video recording is only supported for CNN (pixel-based) environments.")
            print("Continuing with standard human rendering configuration...\n")
            render_mode = "human"

        env = make_vec_env(PIXEL_ENV, n_envs=1, seed=42, env_kwargs={
                        "obs_type": "ram", "render_mode": render_mode})

    return env


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"\nModel not found:\n{args.model}")

    print("=" * 65)
    print("Deep Q-Network Greedy Evaluation")
    print("=" * 65)
    print(f"Policy     : {args.policy.upper()}")
    print(f"Model      : {args.model}")
    print(f"Episodes   : {args.episodes}")
    print(f"Recording  : {args.record if args.policy == 'cnn' else False}")
    print("=" * 65)

    env = build_render_env(args.policy, record=args.record)
    model = DQN.load(args.model, env=env)
    print("\nModel loaded successfully.\n")

    fps_delay = 1 / args.fps
    all_rewards = []

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                # Greedy Action Selection (GreedyQPolicy implementation requirement)
                action, _ = model.predict(obs, deterministic=True)

                # VecEnvs automatically step through vector formats arrays
                obs, reward, dones, info = env.step(action)

                done = dones[0]
                episode_reward += reward[0]
                episode_steps += 1

                # VecVideoRecorder handles rendering internally if recording
                if not (args.record and args.policy == "cnn"):
                    env.render()

                time.sleep(fps_delay)

            all_rewards.append(episode_reward)
            print(
                f"Episode {episode + 1:>2} | Reward = {episode_reward:>6.1f} | Steps = {episode_steps}")

    finally:
        env.close()

    print("\n" + "=" * 65)
    print("Evaluation Summary")
    print("=" * 65)
    print(f"Episodes Played : {len(all_rewards)}")
    print(f"Average Reward  : {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"Best Reward     : {max(all_rewards):.2f}")
    print(f"Worst Reward    : {min(all_rewards):.2f}")

    if args.record and args.policy == "cnn":
        print("\nGameplay video saved in: videos/")
    print("\nDone!")


if __name__ == "__main__":
    main()
