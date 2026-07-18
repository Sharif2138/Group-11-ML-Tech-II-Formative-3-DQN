from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import DQN
import gymnasium as gym
import ale_py
import argparse
import json
import os

# % % writefile train.py


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on an Atari environment.")

    parser.add_argument("--experiment-name", type=str, default="baseline")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5")
    parser.add_argument("--policy", type=str, default="CnnPolicy",
                        choices=["CnnPolicy", "MlpPolicy"])

    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)

    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=10_000)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-fraction", type=float, default=0.10)

    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)

    parser.add_argument("--verbose", type=int, default=0)

    return parser.parse_args()


# Environment Setup
def create_env(env_name, seed, policy):
    if policy == "CnnPolicy":
        env = make_atari_env(env_name, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = make_vec_env(
            env_name,
            n_envs=1,
            seed=seed,
            env_kwargs={"obs_type": "ram"}
        )

    return env


# DQN Model
def create_model(env, args, log_dir):

    model = DQN(
        policy=args.policy,
        env=env,

        learning_rate=args.lr,
        gamma=args.gamma,

        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,

        train_freq=args.train_freq,
        target_update_interval=args.target_update,

        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_fraction,

        verbose=args.verbose,

        tensorboard_log=log_dir,
    )

    return model


# Custom Evaluation Callback
class CompactEvalCallback(EvalCallback):
    def __init__(self, *args, csv_file=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.csv_file = csv_file

        # Create CSV and write header
        if self.csv_file is not None:
            with open(self.csv_file, "w") as f:
                f.write("timesteps,reward,best_reward,exploration_rate\n")

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            print(
                f"[{self.num_timesteps:>7,}] "
                f"Reward: {self.last_mean_reward:>7.2f} | "
                f"Best: {self.best_mean_reward:>7.2f} | "
                f"Exploration: {self.model.exploration_rate:.3f}"
            )

            # Save results to CSV
            if self.csv_file is not None:
                with open(self.csv_file, "a") as f:
                    f.write(
                        f"{self.num_timesteps},"
                        f"{self.last_mean_reward},"
                        f"{self.best_mean_reward},"
                        f"{self.model.exploration_rate}\n"
                    )

        return continue_training

# Training


def train(args):
    base_dir = "/kaggle/working/"

    # assert os.path.isdir("/content/drive/MyDrive"), (
    #     "Google Drive doesn't appear to be mounted. "
    #     "Run drive.mount('/content/drive') in a cell before training."
    # )
    model_dir = os.path.join(base_dir, "models", args.experiment_name)
    log_dir = os.path.join(base_dir, "logs", args.experiment_name, )

    checkpoint_dir = os.path.join(
        base_dir, "checkpoints", args.experiment_name, )

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(
        os.path.join(model_dir, "config.json"),
        "w",
    ) as f:

        json.dump(
            vars(args),
            f,
            indent=4,
        )

    env = create_env(args.env, args.seed, args.policy, )

    eval_env = create_env(args.env, args.seed + 100, args.policy, )

    model = create_model(env, args, log_dir,)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_checkpoint",
    )

    eval_callback = CompactEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        verbose=0,
        n_eval_episodes=10,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback,])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=10,
        )

        model.save(os.path.join(model_dir, "dqn_model", ))

        print("\nTraining completed successfully!")
        print(f"Final model    : {os.path.join(model_dir, 'dqn_model.zip')}")
        print(f"Best model     : {os.path.join(model_dir, 'best_model.zip')}")
        print(f"Config         : {os.path.join(model_dir, 'config.json')}")

    finally:
        env.close()
        eval_env.close()


# Main

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
