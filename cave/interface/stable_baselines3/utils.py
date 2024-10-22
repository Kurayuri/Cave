from typing import Callable, Sequence
import stable_baselines3.common.base_class
import stable_baselines3
import torch
from ..import torch as _torch


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net

    def forward(self, observation):
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden)


def extract_model(algo: stable_baselines3.common.base_class.BaseAlgorithm):
    model = None
    if isinstance(algo, stable_baselines3.DDPG):
        model = algo.policy.actor.mu
    elif isinstance(algo, stable_baselines3.DQN):
        model = algo.policy.q_net.q_net
    elif isinstance(algo, stable_baselines3.PPO):
        model = OnnxablePolicy(algo)
    elif isinstance(algo, stable_baselines3.SAC):
        model = algo.policy.actor.latent_pi
    elif isinstance(algo, stable_baselines3.TD3):
        model = algo.policy.actor.mu
    return model


def export_to_onnx(algo: stable_baselines3.common.base_class.BaseAlgorithm, save_path: str, input_shape: Callable[[Sequence[int]], Sequence[int]] | Sequence[int] | None = None):

    model = extract_model(algo)
    _input_shape = algo.observation_space.shape
    if isinstance(input_shape, Callable):
        _input_shape = input_shape(_input_shape)
    if isinstance(input_shape, Sequence):
        _input_shape = input_shape
    args = torch.randn(_input_shape)

    _torch.export_to_onnx(model, save_path, args)
