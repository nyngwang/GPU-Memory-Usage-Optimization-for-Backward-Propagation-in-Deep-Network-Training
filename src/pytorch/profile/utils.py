from time import sleep
import torch
import gc



def register_forward_pre_hooks(gen_forward_pre_hook, named_modules, types):
    handles = []

    for name, module in named_modules:
        if isinstance(module, tuple(types)):
            handles.append(
                module.register_forward_pre_hook(gen_forward_pre_hook(
                    name=name,
                    type=module.__class__.__name__
                ))
            )

    return handles


def register_forward_hooks(gen_forward_hook, named_modules, types):
    handles = []

    for name, module in named_modules:
        if isinstance(module, tuple(types)):
            handles.append(
                module.register_forward_hook(gen_forward_hook(
                    name=name,
                    type=module.__class__.__name__
                ))
            )

    return handles


def register_backward_pre_hooks(gen_backward_pre_hook, named_modules, types):
    handles = []

    for name, module in named_modules:
        if isinstance(module, tuple(types)):
            handles.append(
                module.register_full_backward_pre_hook(gen_backward_pre_hook(
                    name=name,
                    type=module.__class__.__name__
                ))
            )

    return handles


def register_backward_hooks(gen_backward_hook, named_modules, types):
    handles = []

    for name, module in named_modules:
        if isinstance(module, tuple(types)):
            handles.append(
                module.register_full_backward_hook(gen_backward_hook(
                    name=name,
                    type=module.__class__.__name__
                ))
            )

    return handles
