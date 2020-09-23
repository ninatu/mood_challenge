import math
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn

from anomaly_detection.dpa.pg_networks import ProgGrowNetworks, ProgGrowStageType, STABNetwork, TRNSNetwork, \
    NetworkType
from anomaly_detection.dpa.layers import ConvBlock, get_act_layer, PreActResnetBlock, PreActResnetBlockUp, ConcatLayer, \
    FadeinLayer


class AbstractDecoderNetworks(ProgGrowNetworks, ABC):
    """
    The architecture of network is taken from

    Karras, Tero, et al.
    "Progressive growing of gans for improved quality, stability, and variation."
    arXiv preprint arXiv:1710.10196 (2017).
    """

    def __init__(self, input_res, max_output_res, input_dim, output_dim, inner_dims, normalize_latents=False,
                 norm='none', pad_type='zero', upsample_mode='nearest', mode_3d=False):
        """
        API allows create Decoder which takes tensor as input.
        For example, input tensor may be 16x4x4 (input_res = 4, input_dim=16).
        But input_res must be degree of 2(except for 2): 1, 4, 8, 16, 32, ...

        inner_dims -- list of depths of convolution layers.

        """
        super().__init__()

        self.input_res = input_res
        self.max_output_res = max_output_res
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_dims = inner_dims

        self.norm = norm
        self.pad_type = pad_type
        self.upsample_mode = upsample_mode
        self.mode_3d = mode_3d

        self._create_networks()

    def _create_networks(self):
        res_blocks = OrderedDict()

        # next resolution
        resolution = self._get_first_block_resolution()
        res_blocks['res_{}'.format(resolution)] = self._get_first_block()
        prev_postprocess = self._get_rgb_block(resolution)

        stab_model = nn.Sequential(res_blocks)
        stab_model.add_module('postprocess_res_{}'.format(resolution), prev_postprocess)
        self.set_net(ProgGrowStageType.stab, resolution, STABNetwork(stab_model))

        resolution *= 2
        while resolution <= self.max_output_res:
            # trns dpa
            trns_model = nn.Sequential(res_blocks)

            low_resl = nn.Sequential(OrderedDict([
                ('postprocess_res_{}'.format(resolution // 2), prev_postprocess),
                ('from_res_{}_to_res_{}'.format(resolution // 2, resolution),
                    nn.Upsample(scale_factor=2, mode=self.upsample_mode))
            ]))

            new_stage_block = self._get_intermediate_block(resolution)
            new_post_process = self._get_rgb_block(resolution)

            high_resl = nn.Sequential()
            high_resl.add_module('res_{}'.format(resolution), new_stage_block)
            high_resl.add_module('postprocess_res_{}'.format(resolution), new_post_process)

            trns_model.add_module('concat', ConcatLayer(low_resl, high_resl))
            trns_model.add_module('fadein', FadeinLayer())
            self.set_net(ProgGrowStageType.trns, resolution, TRNSNetwork(trns_model))

            # stab dpa
            res_blocks['res_{}'.format(resolution)] = new_stage_block
            prev_postprocess = new_post_process

            stab_model = nn.Sequential(res_blocks)
            stab_model.add_module('postprocess_res_{}'.format(resolution), prev_postprocess)
            self.set_net(ProgGrowStageType.stab, resolution, STABNetwork(stab_model))
            resolution *= 2

    def _get_num_filters(self, resolution):
        stage = int(math.log(resolution, 2))
        init_stage = int(math.log(self._get_first_block_resolution(), 2))

        return self.inner_dims[stage - init_stage]

    def _get_first_block_resolution(self):
        if self.input_res == 1:
            return 4
        else:
            return self.input_res

    @abstractmethod
    def _get_first_block(self):
        pass

    @abstractmethod
    def _get_intermediate_block(self, resolution):
        pass

    @abstractmethod
    def _get_rgb_block(self, resolution):
        pass


class Resnet9DecoderNetworks(AbstractDecoderNetworks):
    @staticmethod
    def _init_layers(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return model

    def _get_block(self, block, prev_nf, nf):
        return block(prev_nf, nf, norm=self.norm, act='leaky_relu', pad_type=self.pad_type, mode_3d=self.mode_3d)

    def _get_first_block(self):
        resolution = self._get_first_block_resolution()
        nf = self._get_num_filters(resolution)

        if self.input_res == 1:
            layers = get_act_layer('leaky_relu') \
                     + [ConvBlock(self.input_dim, nf, kernel_size=4, stride=1, padding=3, norm='none',
                                 act='linear', pad_type='zero', mode_3d=self.mode_3d)]
        else:
            layers = [self._get_block(PreActResnetBlock, self.input_dim, nf)]

        return self._init_layers(nn.Sequential(*layers))

    def _get_intermediate_block(self, resolution):
        prev_nf = self._get_num_filters(resolution / 2)
        nf = self._get_num_filters(resolution)
        layers = [
            self._get_block(PreActResnetBlockUp, prev_nf, nf),
        ]
        return self._init_layers(nn.Sequential(*layers))

    def _get_rgb_block(self, resolution):
        nf = self._get_num_filters(resolution)
        layers = get_act_layer('leaky_relu')
        layers += [ConvBlock(nf, self.output_dim, kernel_size=3, stride=1, padding=1, norm='none',
                             act='linear', pad_type=self.pad_type, mode_3d=self.mode_3d)]

        return self._init_layers(nn.Sequential(*layers))


class Resnet18DecoderNetworks(Resnet9DecoderNetworks):
    def _get_intermediate_block(self, resolution):
        prev_nf = self._get_num_filters(resolution / 2)
        nf = self._get_num_filters(resolution)
        layers = [
            self._get_block(PreActResnetBlockUp, prev_nf, nf),
            self._get_block(PreActResnetBlock, nf, nf),
        ]
        return self._init_layers(nn.Sequential(*layers))


DECODER_NETWORKS = {
    NetworkType.residual9: Resnet9DecoderNetworks,
    NetworkType.residual18: Resnet18DecoderNetworks,
}