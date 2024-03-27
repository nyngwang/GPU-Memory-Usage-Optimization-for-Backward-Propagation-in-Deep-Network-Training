import os

from .models.vgg import VGG
from .models.alexnet import AlexNet
from .models.utils.test import FakeCmdargs
from .algo.scope import AlgoScope
from .algo.algorithm import *

MODEL: AlgoScope = None
out = []


def file_writer(lines, file_name):
    # write to filepath.
    # use append mode because all entries have to be kept.
    with open('./pytorch/ipynb/{}.txt'.format(file_name), 'a') as file:
        for i, v in enumerate(lines):
            file.write("training phase index={},{}\n".format(i, v))


if __name__ == '__main__':
    print('\n\n')
    print('BEGIN')
    
    if os.path.isfile('./pytorch/ipynb/{}.txt'.format('prediction')):
        os.remove('./pytorch/ipynb/{}.txt'.format('prediction'))

    # create fake cmdargs.
    cmdargs = FakeCmdargs()
    cmdargs.smd = 'chen_et_al'
    cmdargs.smd = 'segment_cost_with_max'
    cmdargs.algo3 = True
    # NOTE: no-profile 就是忽略 Pytorch report 某些資訊。
    cmdargs.np = [ 'c' ]

    # NOTE: uncomment this if you want to draw for AlexNet.
    # cmdargs.alex= True
    # model = AlexNet(
    #     cmdargs=cmdargs,
    #     init_weight=True,
    # )

    model = VGG(
        specs_name='vgg19',
        cmdargs=cmdargs,
        init_weight=True
    )
    print('debug model.checkpoints:', model.checkpoints)

    MODEL = AlgoScope(
        'vgg19',
        model.cost_layer,
        cmdargs
    )

    # assign checkpoints manually.
    # model.algo.checkpoints = [3, 6, 24]

    out = model.algo.make_prediction(model.model_weight, algo2=False)
    print(len(out))

    file_writer(out, 'prediction')


