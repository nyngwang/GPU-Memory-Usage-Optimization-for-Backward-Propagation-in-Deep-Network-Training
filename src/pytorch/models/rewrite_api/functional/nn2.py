from time import sleep
import torch
import pydevd
import gc
from threading import current_thread


def checkpoint(func, inputs, params, use_checkpoint):
    if not use_checkpoint:
        return func(*inputs)

    args = tuple(inputs) + tuple(params)
    return CheckpointFunction.apply(func, len(inputs), *args)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        # print('in forward, current_thread: ', current_thread())
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # print('in backward, current_thread: ', current_thread())
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the Tensor storage in place,
            # which is not allowed for detach()'d Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)

        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=False,
        )

        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        cuda_memory_logger('## backward, new after segment')()

        return (None, None) + input_grads
