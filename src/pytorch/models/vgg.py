import torch
import torch.nn as nn
from pytorch_memlab import MemReporter

from .common import Checkpointable
from .utils.model import get_flatten_pytorch_model, merge_CR
from .utils.profile import should_disable_profile, cuda_memory_logger, get_filename_postfix_by_cmdargs


class VGG(Checkpointable):
    def __init__(self,
        specs_name,
        cmdargs,
        init_weight=True, # `False` on pre-trained.
    ):
        super().__init__(
            specs_name=specs_name,
            cmdargs=cmdargs,
            init_weight=init_weight,
        )

        layers = get_flatten_pytorch_model(self.specs_name)
        self.add_layer_filter(merge_CR)
        for filter in self.filters:
            layers = filter(layers)
        self.layers = nn.Sequential(*layers)

        if len([ layer for layer in self.layers if isinstance(layer, nn.Dropout) ]) > 0:
            self.has_dropout = True
        else:
            self.has_dropout = False

        print('len(self.layers):', len(self.layers))
        self.run_after_self_layers_are_built()
        self.run_algorithm()

        # logging.
        # TODO: move to another base class specifically for profiling.
        self.__enabled_logging = False
        self.handle_layer = list()
        self.handle_layer_fw = list()


    ## public methods.


    def enable_memlab(self):
        self.reporter = MemReporter(self)
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

                # the sane default for VGG.
                if isinstance(layer, nn.Conv2d):
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

            if self.__enabled_logging:
                self.handle_layer_fw.append(segment.register_forward_hook(
                    cuda_memory_logger(
                        '## forward, Segment',
                        file_name=('forward{}' if self.cmdargs.gc else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                    )
                ))
                self.handle_layer.append(segment.register_full_backward_pre_hook(
                    cuda_memory_logger(
                        '## backward, Segment',
                        file_name=('backward{}' if self.cmdargs.gc else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                    )
                ))

