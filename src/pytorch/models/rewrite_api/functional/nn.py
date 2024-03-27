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
        ctx.length = length
        ctx.save_for_backward(*args)

        with torch.no_grad():
            output_tensors = ctx.run_function(*args[:length])
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # print('in backward, current_thread: ', current_thread())
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input_tensors = list(ctx.saved_tensors[:ctx.length])
        input_params = list(ctx.saved_tensors[ctx.length:])
        input_tensors = [x.detach().requires_grad_(True) for x in input_tensors]

        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)

        input_grads = torch.autograd.grad(
            output_tensors,
            input_tensors + input_params,
            output_grads,
            allow_unused=False,
        )

        del input_tensors
        del input_params
        del shallow_copies
        del output_tensors
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        input_grads = (None,)*ctx.length + input_grads[ctx.length:]

        return (None, None) + input_grads
