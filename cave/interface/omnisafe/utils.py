import omnisafe
import torch

from typing import Callable, Sequence


def export_to_onnx(model_name: str, save_path: str, log_dir: str, input_shape: Callable | Sequence[int] | None = None):
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    evaluator.load_saved(save_dir=log_dir, model_name=model_name)

    if evaluator._actor is not None:
        model = evaluator._actor
    elif evaluator._planner is not None:
        raise NotImplementedError
    else:
        raise ValueError('The policy must be provided or created before evaluating the agent.')

    _input_shape = evaluator._env.observation_space.shape
    if isinstance(input_shape, Callable):
        _input_shape = input_shape(_input_shape)
    if isinstance(input_shape, Sequence):
        _input_shape = input_shape
    args = torch.randn(_input_shape)

    from ..import torch
    torch.export_to_onnx(model, save_path, args)
