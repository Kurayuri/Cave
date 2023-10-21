from .Environment import CallBack, maker_Environment
from .import KEYWORD
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from collections import OrderedDict
from typing import Union
import stable_baselines3
import numpy as np
import argparse
import warnings
import torch
import json
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ""
warnings.filterwarnings("ignore")


def maker_TrainTestAPI(**kwargs):
    def func(**kargs):
        api = TrainTestAPI(**kargs, **kwargs)
        return api.ans
    return func


class TrainTestAPI:
    ALGOS = {
        "A2C": stable_baselines3.A2C,
        "DDPG": stable_baselines3.DDPG,
        "DQN": stable_baselines3.DQN,
        "PPO": stable_baselines3.PPO,
        "SAC": stable_baselines3.SAC,
        "TD3": stable_baselines3.TD3
    }

    def __init__(self,
                 env_id: str = None,
                 env_kwargs: dict = {},
                 algo: str = None,
                 algo_kwargs: dict = {},
                 curr_model_dirpath: str = None,
                 next_model_dirpath: str = None,
                 model_filename: str = None,
                 onnx_filename: str = "model.onnx",
                 reward_api: Union[callable, str] = None,
                 test_log_filename: str = "test.log",
                 total_cycle: int = 100,
                 mode: str = None,
                 nproc: int = 1):
        # Init ALGO

        # Initialize path
        assert curr_model_dirpath or next_model_dirpath, 'At least one of <curr_model_dirpath> and <next_model_dirpath> is required.'

        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.algo_kwargs = algo_kwargs
        self.curr_model_dirpath = curr_model_dirpath
        self.next_model_dirpath = next_model_dirpath
        self.model_filename = model_filename
        self.onnx_filename = onnx_filename
        self.reward_api = reward_api
        self.test_log_filename = test_log_filename
        self.total_cycle = total_cycle
        self.mode = mode
        self.nproc = nproc

        self.ALGO = self.ALGOS[algo]

        self.curr_model_path = os.path.join(curr_model_dirpath, model_filename) if curr_model_dirpath else None

        self.next_model_path, self.next_onnx_path = (os.path.join(next_model_dirpath, model_filename),
                                                     os.path.join(next_model_dirpath, onnx_filename)) if next_model_dirpath else (None, None)
        self.ans = None
        if mode == KEYWORD.TRAIN:
            self.train()
        elif mode == KEYWORD.TEST:
            self.test()

    def extract_onnxable_model(self, model):

        onnxable_model = None
        if isinstance(model, stable_baselines3.DDPG):
            onnxable_model = model.policy.actor.mu
        elif isinstance(model, stable_baselines3.DQN):
            onnxable_model = model.policy.q_net.q_net
        elif isinstance(model, stable_baselines3.PPO):
            class OnnxablePolicy(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.extractor = model.policy.mlp_extractor
                    self.action_net = model.policy.action_net
                    self.value_net = model.policy.value_net

                def forward(self, observation):
                    action_hidden, value_hidden = self.extractor(observation)
                    return self.action_net(action_hidden)

            onnxable_model = OnnxablePolicy(model)

        elif isinstance(model, stable_baselines3.SAC):
            onnxable_model = model.policy.actor.latent_pi
        elif isinstance(model, stable_baselines3.TD3):
            onnxable_model = model.policy.actor.mu
        return onnxable_model

    def detect_reward_api(self, reward_api, path: str = ""):
        if isinstance(reward_api, CallBack):
            return reward_api
        if isinstance(reward_api, str):
            reward_api_path = os.path.join(path, reward_api)
            return reward_api_path if os.path.exists(reward_api_path) else reward_api

    def train(self):
        os.makedirs(self.next_model_dirpath, exist_ok=True)

        self.reward_api = self.detect_reward_api(self.reward_api, self.next_model_dirpath)

        
        env = SubprocVecEnv([maker_Environment(self.env_id, self.env_kwargs, self.reward_api, self.next_model_dirpath, rank)
                             for rank in range(self.nproc)], start_method='fork') if self.nproc > 1 else  maker_Environment(
                                 self.env_id, self.env_kwargs, self.reward_api, self.next_model_dirpath)()

        if self.curr_model_path is not None:
            model = self.ALGO.load(self.curr_model_path, env=env, tensorboard_log=self.next_model_dirpath, **self.algo_kwargs)
        else:
            model = self.ALGO(env=env, verbose=0, tensorboard_log=self.next_model_dirpath, **self.algo_kwargs)

        model.learn(total_timesteps=self.total_cycle, callback=CallBack(), progress_bar=True)
        env.reset()

        model.save(self.next_model_path)

        # Export to ONNX
        observation_size = model.observation_space.shape
        dummy_input = torch.randn(1, *observation_size)

        onnxable_model = self.extract_onnxable_model(model)
        torch.onnx.export(
            onnxable_model,
            dummy_input,
            self.next_onnx_path,
            opset_version=9,
            input_names=["input"],
        )

        print(self.next_onnx_path)

    def test(self):
        self.reward_api = self.detect_reward_api(self.reward_api, self.curr_model_dirpath)

        # env = Environment(env_id, env_kwargs, reward_api)
        env = SubprocVecEnv([maker_Environment(self.env_id, self.env_kwargs, self.reward_api, rank=rank)
                             for rank in range(self.nproc)], start_method='fork') if self.nproc > 1 else  maker_Environment(
                                 self.env_id, self.env_kwargs, self.reward_api)()

        model = self.ALGO.load(self.curr_model_path, env=env, **self.algo_kwargs)

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f"Test: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        result = {"mean_reward": mean_reward, "std_reward": std_reward}
        with open(os.path.join(self.curr_model_dirpath, self.test_log_filename), 'w') as f:
            json.dump(result, f)
        self.ans = result
        return result


# %%
def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("API for training and testing.")
    parser.add_argument("--env_id",
                        type=str,
                        required=True,
                        help="ID of gym env.")
    parser.add_argument("--env_kwargs",
                        type=dict,
                        default={},
                        help="Kwargs of gym env.")
    parser.add_argument("--algo",
                        type=str,
                        required=True,
                        help="Agorithm.")
    parser.add_argument("--algo_kwargs",
                        type=dict,
                        default={},
                        help="Kwargs of algorith,.")
    parser.add_argument("--curr_model_dirpath",
                        type=str,
                        default=None,
                        help="Path of the direcotry where exists a pretrained model.")
    parser.add_argument('--next_model_dirpath',
                        type=str,
                        default=None,
                        help="Path of the direcotry to save the model.")
    parser.add_argument('--model_filename',
                        type=str,
                        default="Model",
                        help='Filename of the model.')
    parser.add_argument('--onnx_filename',
                        type=str,
                        default="model.onnx",
                        help='Filename of the onnx model.')
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
                        default="",
                        help='Mode in train or test.')
    parser.add_argument('--nproc',
                        type=int,
                        default=1,
                        help='Number of processes.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_filename = "CartPole"
    env_id = "CatePole-v1"
    algo = "PPO"

    algo_kwargs = OrderedDict([
        ('buffer_size', 200000),
        ('gamma', 0.98),
        ('gradient_steps', -1),
        ('learning_rate', 0.001),
        ('learning_starts', 10000),
        ('action_noise', NormalActionNoise(np.zeros(1), np.ones(1) * 0.1)),
        ('policy', 'MlpPolicy'),
        ('policy_kwargs', dict(net_arch=[400, 300], n_critics=1)),
        ('train_freq', 1)
    ])
    env_kwargs = {}

    TrainTestAPI(env_id=env_id,
                 env_kwargs=env_kwargs,
                 algo=algo,
                 algo_kwargs=algo_kwargs,
                 curr_model_dirpath=args.curr_model_dirpath,
                 next_model_dirpath=args.next_model_dirpath,
                 model_filename=model_filename,
                 onnx_filename=args.onnx_filename,
                 reward_api=args.reward_api,
                 test_log_filename=args.test_log_filename,
                 total_cycle=args.total_cycle,
                 mode=args.mode,
                 nproc=args.nproc
                 )
