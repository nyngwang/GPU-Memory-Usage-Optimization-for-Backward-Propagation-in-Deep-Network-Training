import torch.nn as nn
import torchvision.models as models


def flatten_nn_module(module):
    # what if more nested?
    out = []

    for layer in module:
        if isinstance(layer, nn.ModuleList)\
        or isinstance(layer, nn.Sequential):
            out.extend(flatten_nn_module(layer))
        else:
            out.append(layer)
            # NOTE: pytorch built-in class does not contains Flatten layer.
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                out.append(nn.Flatten())
    return out


def get_flatten_pytorch_model(name_model):

    name_model_lower = name_model.lower()

    if not name_model_lower in models.list_models():
        print('pytorch does not provide provide model {}, return empty list.'.format(name_model))
        return []

    model_class = models.get_model(name_model_lower)

    # NOTE: `model._modules` is a dictionary of all layers.
    return flatten_nn_module(
        [ v for _, v in model_class._modules.items() ]
    )

# NOTE: use ReLU no-inplace
# out = [
#     nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
#     for layer in out
# ]

# NOTE: use dropout-inplace
# RESULT: cannot hook
# out = [
#     nn.Dropout(p=layer.p, inplace=True) if isinstance(layer, nn.Dropout) else layer
#     for layer in out
# ]
def merge_CR(module):
    # NOTE: this function assumes a flatten module.
    out = []
    idx_layer = 0
    end = len(module)

    while idx_layer < end:
        if (
            idx_layer < end-1
        ) and (
            isinstance(module[idx_layer], nn.Conv2d)
            or isinstance(module[idx_layer], nn.Linear)
        ) and (
            isinstance(module[idx_layer+1], nn.ReLU)
        ):
            out += [
                nn.Sequential(
                    module[idx_layer],
                    module[idx_layer+1]
                )
            ]
            idx_layer = idx_layer+2
        else:
            out += [ module[idx_layer] ]
            idx_layer = idx_layer+1

    return out


def map_symbol_to_class(sym):

    # NOTE: algo3 will get better result when 'c' is applied.
    if sym == 'c': return nn.Conv2d
    if sym == 'r': return nn.ReLU
    if sym == 'm': return nn.MaxPool2d
    if sym == 'l': return nn.Linear
    # tools.
    if sym == 'f': return nn.Flatten
    if sym == 'a': return nn.AdaptiveAvgPool2d
    if sym == 'd': return nn.Dropout
    if sym == 'b': return nn.BatchNorm2d

    return None


if __name__ == '__main__':
    seq_nested = nn.Sequential(*[
        nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3),
        # nn.Sequential(*[
        #     nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3),
        #     nn.Sequential(*[
        #         nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3),
        #         nn.Sequential(
        #             nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3),
        #         )
        #     ])
        # ])
    ])

    my_list = [
        nn.Conv2d(in_channels=99, out_channels=100, kernel_size=3),
        nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3),
    ]

    seq_nested.extend(my_list)

    print(seq_nested)


    # out = flatten_nn_module(seq_nested)
    #
    # for layer in out:
    #     print(layer)
    #
    # seq_flatten = nn.Sequential(*out)
    # IDX = 3
    # print('out[{}]: {}'.format(IDX, out[IDX]))

    # print()
    # out = get_flatten_pytorch_model('vgg19')
    #
    # for layer in out:
    #     print(layer)
    # print(out[-8].output_size[0] * out[-8].output_size[1])
