import torch
import torch.nn as nn
from pytorch_memlab import MemReporter

from .common import Checkpointable
from .utils.unet import *
from .utils.profile import should_disable_profile, cuda_memory_logger, get_filename_postfix_by_cmdargs


class Unet(Checkpointable):
    def __init__(self,
        cmdargs,
        init_weight=True, # `False` on pre-trained.
        bilinear=False,
    ):
        super().__init__(
            specs_name=self.__class__.__name__,
            cmdargs=cmdargs,
            init_weight=init_weight,
        )

        self.__build_self_layers(
            n_channels=3,
            n_classes=self.out_classes,
            bilinear=bilinear,
        )
        self.run_after_self_layers_are_built()
        self.run_algorithm()

        # logging.
        # TODO: move to another base class specifically for profiling.
        self.__enabled_logging = False
        self.handle_layer = list()
        self.handle_layer_fw = list()


    ## public methods.

    ### build self.layers.


    def __build_self_layers(self,
        n_channels,
        n_classes,
        bilinear=False,
    ):
        factor = 2 if bilinear else 1

        out = nn.ModuleList([
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor),

            Up(1024, 512 // factor, bilinear),
            Up(512, 256 // factor, bilinear),
            Up(256, 128 // factor, bilinear),
            Up(128, 64, bilinear),
            OutConv(64, n_classes),
        ])
        # flatten.
        out = [
            layer
            for layer in out.modules()
            if not isinstance(layer, (Unet, nn.Sequential, nn.ModuleList, DoubleConv, Down, Up, OutConv))
        ]
        self.layers = out


    ### logging.


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

        # there is no segment when gc is not enabled.
        if not self.cmdargs.gc:
            for layer in self.layers:

                if should_disable_profile(layer, self.cmdargs):
                    continue

                if isinstance(layer, nn.Flatten):
                    continue

                self.handle_layer_fw.append(layer.register_forward_hook(
                    cuda_memory_logger(
                        '## forward, Layer {}'.format(layer.__class__.__name__),
                        file_name=('forward{}' if self.gradient_checkpoint else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                    )
                ))
                self.handle_layer.append(layer.register_full_backward_hook(
                    cuda_memory_logger(
                        '## backward, Layer {}'.format(layer.__class__.__name__),
                        file_name=('backward{}' if self.gradient_checkpoint else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                    )
                ))
            return


        # always treat the last-layer segment specially.
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
                            file_name=('forward{}' if self.gradient_checkpoint else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                        )
                    ))
                    self.handle_layer.append(layer.register_full_backward_hook(
                        cuda_memory_logger(
                            '## backward, Layer {}'.format(layer.__class__.__name__),
                            file_name=('backward{}' if self.gradient_checkpoint else 'no_checkpoint{}').format(get_filename_postfix_by_cmdargs(self.cmdargs))
                        )
                    ))


    ## private methods.


    # TODO: moving this into utils or something.
    def __apply_filter_for_gc(self, module):
        # filter-out some layers before storing the result into self.layers.
        out = module

        # required.

        # NOTE: torch.utils.checkpoint cannot work with inplace-ReLU.
        out = [
            nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in out
        ]
        # out = [
        #     nn.Dropout(p=0.5, inplace=True) if isinstance(layer, nn.Dropout) else layer
        #     for layer in out
        # ]
        return out


