import argparse
import enum
import json
import os
from re import T
import warnings
from collections import OrderedDict
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.base_class
import stable_baselines3.common.off_policy_algorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (EvalCallback, StopTrainingOnMaxEpisodes,
                                                StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback


from cave import Const, Keywords, interface
from cave.Settings import messager
from cave.Environment import CallBack, Environment, make_Environment_fn

os.environ['CUDA_VISIBLE_DEVICES'] = ""
warnings.filterwarnings("ignore")


def create_TrainTestAPI(**kwargs):
    def _creator(**kargs):
        for kw, arg in kwargs.items():
            kargs[kw] = arg
            if kw in kargs:

                messager.warning(f"Warning: {TrainTestAPI.__name__} got multiple values for keyword argument '{kw}'")

        api = TrainTestAPI(**kargs)
        return api.ans
    return _creator


class TrainTestAPIFlag(enum.Flag):
    WANDB = enum.auto()
    TRACEBACK = enum.auto()
    TRAIN = enum.auto()
    TEST = enum.auto()


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
                 env_id: str,
                 env_kwargs: dict = {},
                 env_options: dict = {},
                 algo: str = None,
                 algo_kwargs: dict = {},
                 curr_model_dirpath: str = None,
                 next_model_dirpath: str = None,
                 model_filename: str = "Model.zip",
                 onnx_filename: str = "model.onnx",
                 reward_api: Union[Callable, str] = None,
                 enabled_trace: bool = False,
                 test_log_filename: str = "test.log",
                 total_cycle: Union[int, Tuple[int, str]] = 100,
                 mode: str = None,
                 nproc: int = 1,
                 eval_nproc: int = 1,
                 eval_freq: int = 10000,
                 eval_episode: int = 100,
                 test_episode: int = 100,
                 enabled_wandb: bool = False
                 ):
        # Init ALGO
        # TODO
        self.messager = messager

        # Initialize path
        assert curr_model_dirpath or next_model_dirpath, 'At least one of <curr_model_dirpath> and <next_model_dirpath> is required.'

        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.env_options = env_options
        self.algo_kwargs = algo_kwargs
        self.curr_model_dirpath = curr_model_dirpath
        self.next_model_dirpath = next_model_dirpath
        self.model_filename = model_filename
        self.onnx_filename = onnx_filename
        self.reward_api = reward_api
        self.enabled_traceback = enabled_trace
        self.test_log_filename = test_log_filename
        self.total_timestep = total_cycle
        self.mode = mode
        self.nproc = nproc
        self.eval_nproc = eval_nproc
        self.eval_freq = eval_freq
        self.eval_episode = eval_episode
        self.test_episode = test_episode
        self.enabled_wandb = enabled_wandb

        self.Algo = TrainTestAPI.ALGOS[algo]
        self.buffer_filename = "buffer"

        self.curr_model_path, self.curr_buffer_path = (
            os.path.join(curr_model_dirpath, model_filename),
            os.path.join(curr_model_dirpath, self.buffer_filename)
        )if curr_model_dirpath else (None, None)

        self.next_model_path, self.next_onnx_path, self.next_buffer_path = (
            os.path.join(next_model_dirpath, model_filename),
            os.path.join(next_model_dirpath, onnx_filename),
            os.path.join(next_model_dirpath, self.buffer_filename),
        ) if next_model_dirpath else (None, None, None)

        self.ans = None
        if mode == Keywords.TRAIN:
            self.train()
        elif mode == Keywords.TEST:
            self.test()

    def train(self):
        os.makedirs(self.next_model_dirpath, exist_ok=True)

        self.reward_api = TrainTestAPI.detect_reward_api(
            self.reward_api, self.next_model_dirpath)
        # print(self.reward_api)
        # Use SubprocVecEnv

        if self.nproc > 1:
            log_dirpaths = TrainTestAPI.get_dirpaths(self.next_model_dirpath,
                                                     self.nproc)
            env = SubprocVecEnv([
                make_Environment_fn(self.env_id, self.env_kwargs,
                                    self.env_options, self.reward_api,
                                    log_dirpath) for log_dirpath in log_dirpaths
            ],
                start_method='fork')
        else:
            log_dirpaths = [self.next_model_dirpath]
            env = make_Environment_fn(self.env_id, self.env_kwargs,
                                      self.env_options, self.reward_api,
                                      log_dirpaths[0])()

        model: BaseAlgorithm
        if self.curr_model_path is not None:
            # Resuming Train
            model = self.Algo.load(self.curr_model_path,
                                   env=env,
                                   #    tensorboard_log=self.next_model_dirpath,
                                   tensorboard_log=os.path.dirname(
                                       self.next_model_dirpath),
                                   **self.algo_kwargs)
            if isinstance(model, OffPolicyAlgorithm):
                model.load_replay_buffer(self.curr_buffer_path)
                self.messager.info(f"Load {model.replay_buffer.size()} transitions.", level=Const.INFO)
        else:
            # New Train
            model = self.Algo(env=env,
                              verbose=0,
                              #   tensorboard_log=self.next_model_dirpath,
                              tensorboard_log=os.path.dirname(
                                  self.next_model_dirpath),

                              **self.algo_kwargs)

        # Callback
        eval_env = SubprocVecEnv([make_Environment_fn(self.env_id, self.env_kwargs,
                                                      self.env_options) for i in range(self.eval_nproc)], start_method='fork')
        no_improvement_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=1)
        no_improvement_callback = None

        # reward_threshold_callback = StopTrainingOnRewardThreshold()
        reward_threshold_callback = None
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episode,
            callback_after_eval=no_improvement_callback,
            callback_on_new_best=reward_threshold_callback,
            best_model_save_path=self.next_model_dirpath,
            verbose=1)

        callback = []
        callback.append(eval_callback if self.eval_freq else None)
        if self.enabled_wandb:
            callback.append(WandbCallback(
                # gradient_save_freq=100, model_save_path=f"models/{run.id}",
                verbose=2,
            ))
        # FIX
        # callback.append(CallBack(self.enabled_traceback)
        #                 ) if self.reward_api else None
        callback.append(CallBack(self.enabled_traceback))

        # Learn
        model.learn(total_timesteps=self.total_timestep,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=True)
        obs = env.reset()

        # Gather info, log
        reset_infos = env.reset_infos if isinstance(
            env, SubprocVecEnv) else [obs[1]]

        cave_info = {k: 0 for k in reset_infos[0][Environment.CAVE]}
        for reset_info in reset_infos:
            for k, v in reset_info[Environment.CAVE].items():
                cave_info[k] += v

        cave_info["num_timesteps"] = model.num_timesteps
        print(cave_info)

        TrainTestAPI.gather_log(log_dirpaths, self.next_model_dirpath)

        # Save
        model.save(self.next_model_path)
        if isinstance(model, OffPolicyAlgorithm):
            model.save_replay_buffer(self.next_buffer_path)

        # Export to ONNX

        interface.stable_baselines3.export_to_onnx(model, self.next_onnx_path, lambda shape: (1, *shape))

        print(self.next_onnx_path)

        self.ans = cave_info
        return cave_info

    def test(self):
        assert self.curr_model_dirpath, '<curr_model_dirpath> is required.'

        self.reward_api = TrainTestAPI.detect_reward_api(
            self.reward_api, self.curr_model_dirpath)

        env = SubprocVecEnv(
            [
                make_Environment_fn(self.env_id, self.env_kwargs,
                                    self.env_options, self.reward_api)
                for rank in range(self.nproc)
            ],
            start_method='fork') if self.nproc > 1 else make_Environment_fn(
                self.env_id, self.env_kwargs, self.env_options,
                self.reward_api)()

        model = self.Algo.load(self.curr_model_path,
                               env=env,
                               )

        def evaluate_policy_progress_bar(*args, **kwargs):
            from tqdm.rich import tqdm
            pbar = tqdm(total=self.test_episode)

            def callback(locals, globals):
                pbar.n = len(locals['episode_rewards'])
                pbar.refresh()

            mean_reward, std_reward = evaluate_policy(*args,
                                                      **kwargs,
                                                      callback=callback)
            pbar.n = kwargs['n_eval_episodes']
            pbar.refresh()
            pbar.close()
            return mean_reward, std_reward

        mean_reward, std_reward = evaluate_policy_progress_bar(
            model, env, n_eval_episodes=self.test_episode)

        print(f"Test: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        result = {"mean_reward": mean_reward, "std_reward": std_reward}
        with open(
                os.path.join(self.curr_model_dirpath, self.test_log_filename),
                'w') as f:
            json.dump(result, f)
        self.ans = result
        return result

    @classmethod
    def detect_reward_api(cls, reward_api, path: str = ""):
        if isinstance(reward_api, str):
            reward_api_path = os.path.join(path, reward_api)
            return reward_api_path if os.path.exists(
                reward_api_path) else reward_api
        else:
            return reward_api

    @classmethod
    def get_dirpaths(cls, dirpath: str, nproc: int):
        return [
            os.path.join(dirpath, "rank_%02d" % rank) for rank in range(nproc)
        ]

    @classmethod
    def gather_log(cls, src_dirpaths: Iterable[str], dst_dirpath: str):
        if len(src_dirpaths) == 1:
            if src_dirpaths[0] == dst_dirpath:
                return

        log_filenames = Environment.LOG_FILENAMES.values()
        for log_filename in log_filenames:
            dst_file = open(os.path.join(dst_dirpath, log_filename), 'w')
            for src_dirpath in src_dirpaths:
                try:
                    src_path = os.path.join(src_dirpath, log_filename)
                    with open(src_path, "r") as src_file:
                        dst_file.write(src_file.read())
                except BaseException:
                    messager.warning(f"No such file: {src_path}", level=Const.DEBUG)
            dst_file.close()

    # def gather_info(self, infos):

    @classmethod
    def merge_log(cls, src_dirpaths: Iterable[str], dst_dirpath: str):
        if len(src_dirpaths) == 1:
            if src_dirpaths[0] == dst_dirpath:
                return

        log_filenames = Environment.LOG_FILENAMES.values()
        for log_filename in log_filenames:
            dst_file = open(os.path.join(dst_dirpath, log_filename), 'w')
            src_files = []
            for src_dirpath in src_dirpaths:
                try:
                    src_path = os.path.join(src_dirpath, log_filename)
                    src_files.append(open(src_path, "r"))
                except BaseException:
                    messager.warning(f"No such file: {src_path}", level=Const.DEBUG)

            counter = 1
            while counter:
                counter = 0
                for src_file in src_files:
                    line = src_file.readline()
                    if line:
                        counter += 1
                        dst_file.write(line)

            dst_file.close()
            for src_file in src_files:
                src_files.close()
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
    parser.add_argument("--test_eposide",
                        type=int,
                        default=100,
                        help="Total number of eposides to be test.")
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
                 nproc=args.nproc,
                 test_episode=args.test_episode
                 )
