import gc
import torch
import torch.nn as nn


def cuda_memory_logger(str_prefix, file_name=None, use_reporter=False, reporter=None):
    def hook(*_):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_usage_MiB = torch.cuda.memory_allocated() / 1024**2

        print("{}: {} MB".format(
            str_prefix,
            memory_usage_MiB,
        ))
        if use_reporter:
            reporter.report(verbose=True)

        # write to filepath.
        # use append mode because all entries have to be kept.
        with open('./pytorch/ipynb/{}.txt'.format(file_name), 'a') as file:

            file.write("{},{}\n".format(str_prefix, memory_usage_MiB))

    return hook


def get_filename_postfix_by_cmdargs(cmdargs):
    out = ''
    if cmdargs.arch:       out += '_{}'.format(cmdargs.arch)
    if cmdargs.batch_size: out += '_b{}'.format(cmdargs.batch_size)
    # can return if checkpoint technique is not used.
    if not cmdargs.gc:
        return out
    else:
        out += '_gc'
    if cmdargs.smd:        out += '_{}'.format(cmdargs.smd)
    if cmdargs.cp:         out += '_cp{}'.format(cmdargs.cp)

    return out


def should_disable_profile(layer, cmdargs):
    if not cmdargs.np:
        # NOTE: 隱藏不會建立 additional memory 的層
        return (
            isinstance(layer, nn.Flatten)
            or isinstance(layer, nn.AdaptiveAvgPool2d)
        )

    if 'c' in cmdargs.np and isinstance(layer, nn.Conv2d): return True
    if 'r' in cmdargs.np and isinstance(layer, nn.ReLU): return True
    if 'm' in cmdargs.np and isinstance(layer, nn.MaxPool2d): return True
    if 'l' in cmdargs.np and isinstance(layer, nn.Linear): return True
    if 'd' in cmdargs.np and isinstance(layer, nn.Dropout): return True
    if 'b' in cmdargs.np and isinstance(layer, nn.BatchNorm2d): return True
    # layers that do not use duplicate memory.
    if 'f' in cmdargs.np and isinstance(layer, nn.Flatten): return True
    if 'a' in cmdargs.np and isinstance(layer, nn.AdaptiveAvgPool2d): return True

    return False


