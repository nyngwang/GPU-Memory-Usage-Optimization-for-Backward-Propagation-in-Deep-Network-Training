import torch
import torch.nn as nn
from pytorch_memlab import MemReporter

from .common import Checkpointable

from .utils.model import get_flatten_pytorch_model, merge_CR
from .utils.profile import should_disable_profile, cuda_memory_logger, get_filename_postfix_by_cmdargs


class AlexNet(Checkpointable):
    def __init__(self,
        cmdargs,
        init_weight=True, # `False` on pre-trained.
    ):
        super().__init__(
            specs_name=self.__class__.__name__,
            cmdargs=cmdargs,
            init_weight=init_weight,
        )
        # TODO: should use another key to be more reasonable.
        self.dataset['image_width'] = 55

        layers = get_flatten_pytorch_model(self.specs_name)
        self.add_layer_filter(merge_CR)
        for filter in self.filters:
            layers = filter(layers)
        self.layers = nn.Sequential(*layers)
        
        if len([ layer for layer in self.layers if isinstance(layer, nn.Dropout) ]) > 0:
            self.has_dropout = True
        else:
            self.has_dropout = False
        self.run_after_self_layers_are_built()
        self.run_algorithm()

        # logging.
        self.__enabled_logging = False
        self.handle_layer = list()
        self.handle_layer_fw = list()



    ## public methods.


    def __apply_filters(self, module):
        # filter-out some layers before storing the result into self.layers.
        out = module

        # required.

        # NOTE: torch.utils.checkpoint cannot work with inplace-ReLU.
        # out = [
        #     nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        #     for layer in out
        # ]
        return out


    def __merge_CR(self, module):

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


    def enable_memlab(self):
        self.reporter = MemReporter(self)
        # TODO: should I keep this?
        for seg in self.segments:
            seg.reporter = self.reporter


    def enabled_logging(self, enabled, fw=True):
        self.__enabled_logging = enabled

        if not enabled and not fw:
            for h in self.handle_layer_fw: h.remove()
            return

        if not enabled:
            for h in self.handle_layer_fw: h.remove()
            for h in self.handle_layer: h.remove()
            return

        for segment in self.segments:
            for layer in segment:

                if should_disable_profile(layer, self.cmdargs):
                    continue

                if isinstance(layer, nn.Flatten):
                    continue

                if self.__enabled_logging:
                    self.handle_layer_fw.append(layer.register_forward_hook(
                        cuda_memory_logger(
                            '## forward, Layer {}'.format(layer.__class__.__name__),
                            file_name=('forward{}' if self.cmdargs.gc else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                        )
                    ))
                    self.handle_layer.append(layer.register_full_backward_hook(
                        cuda_memory_logger(
                            '## backward, Layer {}'.format(layer.__class__.__name__),
                            file_name=('backward{}' if self.cmdargs.gc else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                        )
                    ))


