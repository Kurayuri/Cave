from .Environment import CallBack, maker_Environment, Environment
from .import KEYWORD
from . import util
from . import CONSTANT
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnMaxEpisodes, StopTrainingOnRewardThreshold
from collections import OrderedDict
from typing import Any, Union, Iterable, Callable, Tuple
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
    def _maker_(**kargs):
        api = TrainTestAPI(**kargs, **kwargs)
        return api.ans
    return _maker_


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
                 env_options: dict = {},
                 algo: str = None,
                 algo_kwargs: dict = {},
                 curr_model_dirpath: str = None,
                 next_model_dirpath: str = None,
                 model_filename: str = "Model.zip",
                 onnx_filename: str = "model.onnx",
                 reward_api: Union[callable, str] = None,
                 test_log_filename: str = "test.log",
                 total_cycle: Union[int, Tuple[int, str]] = 100,
                 mode: str = None,
                 nproc: int = 1,
                 test_episode: int = 100):
        # Init ALGO

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
        self.test_log_filename = test_log_filename
        self.total_timestep = total_cycle
        self.mode = mode
        self.nproc = nproc
        self.test_episode = test_episode

        self.ALGO = self.ALGOS[algo]

        self.curr_model_path = os.path.join(
            curr_model_dirpath, model_filename) if curr_model_dirpath else None

        self.next_model_path, self.next_onnx_path = (
            os.path.join(next_model_dirpath, model_filename),
            os.path.join(next_model_dirpath,
                         onnx_filename)) if next_model_dirpath else (None,
                                                                     None)
        self.ans = None
        if mode == KEYWORD.TRAIN:
            self.train()
        elif mode == KEYWORD.TEST:
            self.test()

    def train(self):
        os.makedirs(self.next_model_dirpath, exist_ok=True)

        self.reward_api = self.__class__.detect_reward_api(
            self.reward_api, self.next_model_dirpath)

        # Use SubprocVecEnv
        if self.nproc > 1:
            log_dirpaths = self.__class__.get_dirpaths(self.next_model_dirpath,
                                                       self.nproc)
            env = SubprocVecEnv([
                maker_Environment(self.env_id, self.env_kwargs,
                                  self.env_options, self.reward_api,
                                  log_dirpath) for log_dirpath in log_dirpaths
            ],
                                start_method='fork')
        else:
            log_dirpaths = [self.next_model_dirpath]
            env = maker_Environment(self.env_id, self.env_kwargs,
                                    self.env_options, self.reward_api,
                                    log_dirpaths[0])()

        # Resuming Train
        if self.curr_model_path is not None:
            model = self.ALGO.load(self.curr_model_path,
                                   env=env,
                                   tensorboard_log=self.next_model_dirpath,
                                   **self.algo_kwargs)
        else:
            model = self.ALGO(env=env,
                              verbose=0,
                              tensorboard_log=self.next_model_dirpath,
                              **self.algo_kwargs)

        # Callback
        eval_env = maker_Environment(self.env_id, self.env_kwargs,
                                     self.env_options)()
        no_improvement_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=1)
        no_improvement_callback = None
        
        # reward_threshold_callback = StopTrainingOnRewardThreshold()
        reward_threshold_callback = None
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=self.total_timestep // self.nproc // 10,
            callback_after_eval=no_improvement_callback,
            callback_on_new_best=reward_threshold_callback,
            best_model_save_path=self.next_model_dirpath,
            verbose=1)

        callback = [eval_callback]
        callback.append(CallBack()) if self.reward_api else None

        # Learn
        model.learn(total_timesteps=self.total_timestep,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=True)
        obs = env.reset()

        # Gather info, log
        reset_infos = env.reset_infos if isinstance(env,SubprocVecEnv) else [obs[1]]

        info = {k:0 for k in reset_infos[0]}
        for reset_info in reset_infos:
            for k,v in reset_info.items():
                info[k] += v

        info["num_timesteps"] = model.num_timesteps
        print(info)
        
        self.__class__.gather_log(log_dirpaths, self.next_model_dirpath)


        # Save
        model.save(self.next_model_path)

        # Export to ONNX
        observation_size = model.observation_space.shape
        dummy_input = torch.randn(1, *observation_size)

        onnxable_model = self.__class__.extract_onnxable_model(model)
        torch.onnx.export(
            onnxable_model,
            dummy_input,
            self.next_onnx_path,
            opset_version=9,
            input_names=["input"],
        )

        print(self.next_onnx_path)

        self.ans = info
        return info

    def test(self):
        assert self.curr_model_dirpath, '<curr_model_dirpath> is required.'

        self.reward_api = self.__class__.detect_reward_api(
            self.reward_api, self.curr_model_dirpath)

        env = SubprocVecEnv(
            [
                maker_Environment(self.env_id, self.env_kwargs,
                                  self.env_options, self.reward_api)
                for rank in range(self.nproc)
            ],
            start_method='fork') if self.nproc > 1 else maker_Environment(
                self.env_id, self.env_kwargs, self.env_options,
                self.reward_api)()

        model = self.ALGO.load(self.curr_model_path,
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
    def extract_onnxable_model(cls, model):
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
                    util.log(f"No such file: {src_path}", level=CONSTANT.DEBUG)
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
                    util.log(f"No such file: {src_path}", level=CONSTANT.DEBUG)
            
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
