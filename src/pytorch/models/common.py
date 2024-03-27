import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .utils.model import map_symbol_to_class
from ..algo.algorithm import Algorithm
from ..algo.scope import AlgoScope


class Checkpointable(nn.Module):
    def __init__(self,
        specs_name,
        cmdargs,
        init_weight,
    ):
        super().__init__()
        self.specs_name = specs_name
        self.init_weight = init_weight
        self.__add_self_cmdargs(cmdargs)
        self.__add_self_dataset()

        # NOTE: a subclass should build the self.layers themselves.


    ## public methods.


    def run_after_self_layers_are_built(self):
        self.__add_self_cost_layer()
        self.__add_self_model_weight()
        if self.init_weight:
            self.__init_weights()


    def run_algorithm(self):
        # run algorithm and create Segments.
        self.segments: nn.Sequential = nn.Sequential()
        self.__run_algorithm()
        print('self.segments have been built')


    def add_layer_filter(self, f):
        if not hasattr(self, 'filters'):
            self.filters = list()

        if not callable(f):
            return
        self.filters.append(f)


    # if you recreate nn.Sequential here you will get an error: https://stackoverflow.com/q/59013109.
    def forward(self, x):

        def run_fn(segment, x):
            return segment(x)

        for segment in self.segments:
            if self.cmdargs.gc:
                x = checkpoint(run_fn, segment, x,
                               use_reentrant=False, preserve_rng_state=self.has_dropout)
            else:
                x = segment(x)

        return x


    ## private methods.


    ### prepareing self.


    def __add_self_cost_layer(self):
        # MATH:
        # * Conv2d: batch_size * ( h * w * output-channels )
        # * Linear: batch_size * out_features

        if self.batch_size <= 0:
            raise RuntimeError('preparing scope for {}: batch_size should be a positive integer.'.format(self.specs_name))

        # TODO: get this from pytorch api `element_size`?
        SIZE_POINT = 4

        # NOTE: we do not prepend an zero here, so algorithm should do the 1-index transformation itself.
        self.cost_layer = []
        # NOTE: we're calculating the cost of layer "output", so this should be renamed to first output height/width.
        h_img = w_img = self.dataset['image_width']

        # TODO: should use the size of the real output by running the layer.
        for layer in self.layers:
            # NOTE: should skip only inplace layers.

            # NOTE: since now we merge CR into a nn.Sequential.
            prev = None
            if isinstance(layer, nn.Sequential):
                if len(layer) > 1:
                    prev = layer[-2]
                layer = layer[-1]

            if isinstance(layer, nn.Conv2d):
                # WARN: this might cause a problem if a Conv2d might shrink its input image.
                last_out_channels = layer.out_channels
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, layer, last_out_channels)
                ]
            elif isinstance(layer, nn.ReLU):
                # NOTE: since we don't profile convolution layers, at least for now.
                if isinstance(prev, nn.Conv2d):
                    last_out_channels = prev.out_channels
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, prev, last_out_channels)
                    if prev else
                    self.cost_layer[-1]
                ]
            elif isinstance(layer, nn.MaxPool2d):
                h_img = h_img//2
                w_img = w_img//2
                self.cost_layer += [
                    # WARN: this might cause a problem if we're in nn.Sequential.
                    self.__compute_cost_layer(h_img, w_img, prev, last_out_channels) / 4
                    if prev else
                    self.cost_layer[-1] / 4
                ]
            elif isinstance(layer, nn.Linear):
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, layer, last_out_channels)
                ]
            elif isinstance(layer, nn.Dropout):
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, prev, last_out_channels)
                    if prev else
                    self.cost_layer[-1]
                ]
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, layer, last_out_channels)
                ]
            elif isinstance(layer, nn.Flatten):
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, prev, last_out_channels)
                    if prev else
                    self.cost_layer[-1]
                ]
            elif isinstance(layer, nn.BatchNorm2d):
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, prev, last_out_channels)
                    if prev else
                    self.cost_layer[-1]
                ]
            elif isinstance(layer, nn.ConvTranspose2d):
                # NOTE: this only works for kernel_size=2, stride=2 in Unet.
                h_img = h_img*2
                w_img = w_img*2
                self.cost_layer += [
                    self.__compute_cost_layer(h_img, w_img, layer, last_out_channels)
                ]
            else:
                raise RuntimeError('wtf is {}'.format(layer))

        # to bytes.
        self.cost_layer = [ cost*SIZE_POINT for cost in self.cost_layer ]
        # to MiB.
        self.cost_layer = [ cost/1024**2 for cost in self.cost_layer ]

        print('len(self.cost_layer):', len(self.cost_layer))
        print('self.cost_layer:', self.cost_layer)


    def __compute_cost_layer(self, h_img, w_img, layer, last_out_channels=0):
        if isinstance(layer, nn.Conv2d):
            return self.batch_size * h_img * w_img * layer.out_channels
        elif isinstance(layer, nn.ConvTranspose2d):
            return self.batch_size * h_img * w_img * layer.out_channels
        elif isinstance(layer, nn.Linear):
            return self.batch_size * layer.out_features
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            return self.batch_size * layer.output_size[0] * layer.output_size[1] * last_out_channels
        else:
            raise RuntimeError('wtf is {}'.format(layer))


    def __add_self_model_weight(self):
        # MATH:
        # * Conv2d: size_kernel^2 * in_channels * out_channels.
        # * Linear: in_features * out_features.
        #   * it requires to flatten all output-channels of the last convolution layer.
        # * MaxPool: 0.

        self.weights = []

        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                weight_ = 0
                for layer_ in layer:
                    if not hasattr(layer_, 'weight'):
                        continue
                    data_point_count = layer_.weight.numel()
                    weight_ += data_point_count * layer_.weight.element_size()
                self.weights += [weight_]
                continue

            if not hasattr(layer, 'weight'):
                self.weights += [ 0 ]
                continue

            data_point_count = layer.weight.numel()
            self.weights += [
                data_point_count * layer.weight.element_size()
            ]

        # to MiB
        self.weights = [ w / 1024**2 for w in self.weights ]
        print('len(weights):', len(self.weights))
        print('self.weights:', self.weights)
        print('sum(self.weights):', sum(self.weights))

        self.model_weight = sum(self.weights)


    ### assign attributes to self.


    def __add_self_cmdargs(self, cmdargs):
        # for convenience of using cmdline argument.

        self.cmdargs = cmdargs

        self.batch_size = cmdargs.batch_size
        # TODO: update the last layer of pytorch built-in model, which uses 1000 classes.
        self.out_classes = cmdargs.out_classes
        self.dataset_name = cmdargs.data.split('/')[-1]
        print("You're training {} on dataset {}".format(self.specs_name, self.dataset_name))
        self.use_batch_norm = cmdargs.bn
        print("segment memory definition: {}".format(cmdargs.smd))


    def __add_self_dataset(self):
        # dataset related constants.

        self.dataset = {}

        if self.dataset_name == 'imagenet'          : self.dataset['image_width'] = 224
        if self.dataset_name == 'tiny-imagenet-200' : self.dataset['image_width'] = 64


    ### algorithm related.


    def __run_algorithm(self):
        # NOTE: each model need to fill-out this form before running the algorithm.
        scope = AlgoScope(
            self.specs_name,
            self.cost_layer,
            self.cmdargs
        )

        # NOTE: algo should be {model,dataset}-agnostic: every model should implement its own preparing.
        self.algo = Algorithm(scope)
        self.algo.solve(self.algo.T_bottom_up_mono)
        self.checkpoints = self.algo.checkpoints
        print('self.checkpoints:', self.checkpoints)

        # the output of algorithm is a list of 1-start indexes of all pivot indexes.
        index_seg_start = 0
        for idx_checkpoint in self.checkpoints[:-1]:
            self.__create_segment(index_seg_start, idx_checkpoint)
            index_seg_start = idx_checkpoint

        # create the last segment.
        self.__create_segment(index_seg_start, len(self.layers))


    def __create_segment(self, start, end):
        if start == end:
            print('DEBUG: we got one 0-length segment omitted.')
            return
        
        segment = self.layers[start:end]
        # print('segment created:', segment)

        self.segments.append(nn.Sequential(*segment))


    ### procedures.


    def __init_weights(self):
        # ref: https://github.com/minar09/VGG16-PyTorch/blob/b921e7fff2449c3523fba319f99be310f750afda/vgg.py#L46
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


