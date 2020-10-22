import torch
from abc import ABC, abstractmethod

from mood.dpa.rec_losses import L2Loss, L1Loss, ReconstructionLossType, PerceptualLoss
from mood.dpa.pg_networks import ProgGrowStageType


class AbstractPGLoss(torch.nn.Module, ABC):
    def __init__(self, max_resolution, mode_3d=False):
        super().__init__()

        self._resolution = max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0
        self._mode_3d = mode_3d

    @abstractmethod
    def set_stage_resolution(self, stage, resolution):
        pass

    @abstractmethod
    def set_progress(self, progress):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def set_reduction(self, reduction):
        pass


class PGPerceptualLoss(AbstractPGLoss):
    def __init__(self, max_resolution, weights_per_resolution, reduction='mean',
                 use_smooth_pg=False,
                 use_L1_norm=False, use_relative_error=False,
                 pad_type='zero',
                 path_to_vgg19_weights=None,
                 normalize_to_vgg_input=True,
                 imagenet_pretrained=False,
                 mode_3d=False,
                 mode_3d_apply_along_axes=[0]):
        super(PGPerceptualLoss, self).__init__(max_resolution, mode_3d)

        self._max_resolution = max_resolution
        self._weights_per_resolution = weights_per_resolution
        self._use_smooth_pg = use_smooth_pg
        self._reduction = reduction
        self._loss = PerceptualLoss(reduction=reduction,
                                    use_L1_norm=use_L1_norm,
                                    use_relative_error=use_relative_error,
                                    pad_type=pad_type,
                                    path_to_vgg19_weights=path_to_vgg19_weights,
                                    imagenet_pretrained=imagenet_pretrained,
                                    normalize_to_vgg_input=normalize_to_vgg_input,
                                    mode_3d=mode_3d,
                                    mode_3d_apply_along_axes=mode_3d_apply_along_axes)

        self._resolution = self._max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0

    def set_stage_resolution(self, stage, resolution):
        self._stage = stage
        self._resolution = resolution
        self._progress = 0

    def set_progress(self, progress):
        self._progress = progress

    def set_reduction(self, reduction):
        self._reduction = reduction
        self._loss.reduction = reduction

    def forward(self, x, y):
        self._loss.set_new_weights(**self._weights_per_resolution[self._resolution])
        loss = self._loss(x, y)

        if self._use_smooth_pg:
            if self._stage == ProgGrowStageType.trns and self._progress < 1:
                prev_res = int(self._resolution / 2)
                self._loss.set_new_weights(**self._weights_per_resolution[prev_res])

                x = torch.nn.functional.upsample(x, scale_factor=0.5, mode='bilinear')
                y = torch.nn.functional.upsample(y, scale_factor=0.5, mode='bilinear')

                prev_loss = self._loss(x, y)
                loss = (1 - self._progress) * prev_loss + self._progress * loss

        return loss


class PGRelativePerceptualL1Loss(PGPerceptualLoss):
    def __init__(self, max_resolution, weights_per_resolution, reduction='mean',
                 use_smooth_pg=False, pad_type='zero', mode_3d=False,
                 mode_3d_apply_along_axes=[0],
                 path_to_vgg19_weights=None,
                 normalize_to_vgg_input=True,
                 imagenet_pretrained=False):
        super().__init__(max_resolution, weights_per_resolution, reduction=reduction,
                         use_smooth_pg=use_smooth_pg,
                         use_L1_norm=True, use_relative_error=True,
                         path_to_vgg19_weights=path_to_vgg19_weights,
                         imagenet_pretrained=imagenet_pretrained,
                         normalize_to_vgg_input=normalize_to_vgg_input,
                         pad_type=pad_type, mode_3d=mode_3d,
                         mode_3d_apply_along_axes=mode_3d_apply_along_axes)


class PGL2Loss(AbstractPGLoss):
    def __init__(self, max_resolution, reduction='mean', mode_3d=False):
        super().__init__(max_resolution)
        self._loss = L2Loss(reduction=reduction, mode_3d=mode_3d)

    def set_stage_resolution(self, stage, resolution):
        pass

    def set_progress(self, progress):
        pass

    def set_reduction(self, reduction):
        self._loss.set_reduction(reduction)

    def forward(self, x, y):
        return self._loss(x, y)


class PGL1Loss(AbstractPGLoss):
    def __init__(self, max_resolution, reduction='mean', mode_3d=False):
        super().__init__(max_resolution)
        self._loss = L1Loss(reduction=reduction, mode_3d=mode_3d)

    def set_stage_resolution(self, stage, resolution):
        pass

    def set_progress(self, progress):
        pass

    def set_reduction(self, reduction):
        self._loss.set_reduction(reduction)

    def forward(self, x, y):
        return self._loss(x, y)


PG_RECONSTRUCTION_LOSSES = {
    ReconstructionLossType.perceptual: PGPerceptualLoss,
    ReconstructionLossType.relative_perceptual_L1: PGRelativePerceptualL1Loss,
    ReconstructionLossType.l1: PGL1Loss,
    ReconstructionLossType.l2: PGL2Loss,
}



