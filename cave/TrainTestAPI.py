import warnings
import time
import argparse
import os
import json
import numpy as np
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

import torch
from collections import OrderedDict
from .Environment import Environment
from typing import Union

os.environ['CUDA_VISIBLE_DEVICES'] = ""
warnings.filterwarnings("ignore")

class TrainTestAPI:
    ALGOS = {
            "DDPG": stable_baselines3.DDPG,
            "DQN": stable_baselines3.DQN,
            "PPO": stable_baselines3.PPO,
            "SAC": stable_baselines3.SAC,
            "TD3": stable_baselines3.TD3
    }
    def __init__(self,
        env_name: str = None,
        algo: str = None,
        algo_kwargs: dict = {},
        model_filename: str = None,
        curr_model_dirpath: str = None,
        next_model_dirpath: str = None,
        onnx_filename: str = "model.onnx",
        reward_api: Union[callable, str] = None,
        test_log_filename: str = "test.log",
        total_cycle: int = 100,
        mode: str = "train",
    ):
        # Init ALGO

        ALGO = self.ALGOS[algo]

        # Initialize path
        if curr_model_dirpath is None and next_model_dirpath is None:
            raise IOError('At least one of <curr_model_dirpath> and <next_model_dirpath> is required.')

        if curr_model_dirpath is not None:
            curr_model_path = os.path.join(curr_model_dirpath, model_filename)
        else:
            curr_model_path = None

        if next_model_dirpath is not None:
            next_onnx_path = os.path.join(next_model_dirpath, onnx_filename)
            next_model_path = os.path.join(next_model_dirpath, model_filename)
        else:
            next_onnx_path = None
            next_model_path = None

        # %% Train
        if mode == "train":
            if reward_api:
                reward_api = os.path.join(next_model_dirpath, reward_api)
            else:
                reward_api = None

            env = Environment(env_name, reward_api, log_dirpath=next_model_dirpath)

            if curr_model_path is not None:
                model = ALGO.load(curr_model_path, env=env, **algo_kwargs)
            else:
                model = ALGO(env=env, verbose=0, **algo_kwargs)

            model.learn(total_timesteps=total_cycle)

            model.save(next_model_path)

            # Export to ONNX
            observation_size = model.observation_space.shape
            dummy_input = torch.randn(1, *observation_size)

            onnxable_model = self.extract_onnxable_model(model)
            torch.onnx.export(
                onnxable_model,
                dummy_input,
                next_onnx_path,
                opset_version=9,
                input_names=["input"],
            )

            print(next_onnx_path)
        # %% Test
        else:
            if reward_api:
                reward_api = os.path.join(curr_model_dirpath, reward_api)
            else:
                reward_api = None

            env = Environment(env_name, reward_api)

            model = ALGO.load(curr_model_path, env=env, **algo_kwargs)

            mean_reward, std_reward = evaluate_policy(model,
                                                    env,
                                                    n_eval_episodes=100)
            print(f"Test: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

            with open(os.path.join(curr_model_dirpath, test_log_filename), 'w') as f:
                json.dump({"mean_reward": mean_reward, "std_reward": std_reward}, f)

    def extract_onnxable_model(self,model):
            onnxable_model = None
            if isinstance(model, stable_baselines3.DDPG):
                onnxable_model = model.policy.actor.mu
            elif isinstance(model, stable_baselines3.DQN):
                onnxable_model = model.policy.q_net.q_net
            elif isinstance(model, stable_baselines3.PPO):
                onnxable_model = model.policy.mlp_extractor.policy_net
            elif isinstance(model, stable_baselines3.SAC):
                onnxable_model = model.policy.actor.latent_pi
            elif isinstance(model, stable_baselines3.TD3):
                onnxable_model = model.policy.actor.mu
            return onnxable_model
# %%


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("API for training and testing.")

    parser.add_argument("--curr_model_dirpath",
                        type=str,
                        default=None,
                        help="Path of the direcotry where exists a pretrained model.")
    parser.add_argument('--next_model_dirpath',
                        type=str,
                        default=None,
                        help="Path of the direcotry to save the model.")
    parser.add_argument('--onnx_filename',
                        type=str,
                        default="model.onnx",
                        help='Filename of the model')
    parser.add_argument('--reward_api',
                        type=str,
                        default=None,
                        help='Reward API')
    parser.add_argument('--test_log_filename',
                        type=str,
                        default="test.log",
                        help='Reward API')
    parser.add_argument("--total_cycle",
                        type=int,
                        default=100,
                        help="Total number of cycles to be trained.")
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='Mode in train or test.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_filename = "Aurora"
    env_name = "NetworkCC-v0"
    algo="PPO"

    algo_kwargs = OrderedDict(
        [('buffer_size', 200000),
         ('gamma', 0.98),
         ('gradient_steps', -1),
         ('learning_rate', 0.001),
         ('learning_starts', 10000),
         #  ('n_timesteps', 20000),
         #  ('noise_std', 0.1),
         #  ('noise_type', 'normal'),
         ('action_noise', NormalActionNoise(np.zeros(1), np.ones(1) * 0.1)),
         ('policy', 'MlpPolicy'),
         #  ('policy_kwargs', 'dict(net_arch=[400, 300])'),
         ('policy_kwargs', dict(net_arch=[400, 300], n_critics=1)),
         ('train_freq', 1)
         #  ('normalize', False)
         ])
    algo_kwargs={'policy': 'MlpPolicy'}

    TrainTestAPI(
        env_name=env_name,
        algo=algo,
        algo_kwargs=algo_kwargs,
        model_filename=model_filename,
        curr_model_dirpath=args.curr_model_dirpath,
        next_model_dirpath=args.next_model_dirpath,
        onnx_filename=args.onnx_filename,
        reward_api=args.reward_api,
        test_log_filename=args.test_log_filename,
        total_cycle=args.total_cycle,
        mode=args.mode
    )
