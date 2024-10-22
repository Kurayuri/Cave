import torch
from typing import Tuple, Any


def get_input_size(model: torch.nn.Module):
    for name, module in model.named_modules():
        if name != '' and hasattr(module, 'weight'):
            first_layer_name = name
            return first_layer_name.weight.shape
    return None


def export_to_onnx(model: torch.nn.Module, save_path: str, args: Tuple[Any, ...] | torch.Tensor | None = None, **kwargs):
    kwargs['opset_version'] = 9
    kwargs['input_names'] = ['input']
    if args is None:
        args = torch.randn(1)

    torch.onnx.export(
        model,
        args,
        save_path,
        **kwargs
    )
