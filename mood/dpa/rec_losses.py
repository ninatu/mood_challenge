from collections import OrderedDict
from enum import Enum
import torch
from torch import nn
from torch.nn import functional as F

from mood.dpa.feature_extractor import PretrainedVGG19FeatureExtractor


class L2Loss(nn.Module):
    def __init__(self, reduction='mean', mode_3d=False):
        super(L2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction
        self._mode_3d = mode_3d

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, y):
        loss = (x - y) * (x - y)

        if not self._mode_3d:
            loss = loss.sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        else:
            loss = loss.sum(4).sum(3).sum(2).sum(1) / (
                        loss.size(1) * loss.size(2) * loss.size(3) * loss.size(4))

        if self._reduction == 'none':
            return loss
        elif self._reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class L1Loss(nn.Module):
    def __init__(self, reduction='none', mode_3d=False):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction
        self._mode_3d = mode_3d

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction

    def forward(self, x, y):
        loss = torch.abs(x - y)
        if self._reduction == 'pixelwise':
            return loss

        if not self._mode_3d:
            loss = loss.sum(3).sum(2).sum(1) / (loss.size(1) * loss.size(2) * loss.size(3))
        else:
            loss = loss.sum(4).sum(3).sum(2).sum(1) / (loss.size(1) * loss.size(2) * loss.size(3) * loss.size(4))

        if self._reduction == 'none':
            return loss
        elif self._reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class PerceptualLoss(torch.nn.Module):
    def __init__(self, reduction='mean', img_weight=0, feature_weights=None,
                 use_L1_norm=False, use_relative_error=False,
                 pad_type='zero',
                 path_to_vgg19_weights=None,
                 imagenet_pretrained=False,
                 normalize_to_vgg_input=True,
                 mode_3d=False, mode_3d_apply_along_axes=[0]):
        super(PerceptualLoss, self).__init__()
        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean', 'pixelwise']

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        self.reduction = reduction

        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error
        self.mode_3d = mode_3d
        self.mode_3d_apply_along_axes = mode_3d_apply_along_axes
        self.normalize_to_vgg_input = normalize_to_vgg_input

        self.model = PretrainedVGG19FeatureExtractor(
            pad_type=pad_type,
            path_to_vgg19_weights=path_to_vgg19_weights,
            pretrained=imagenet_pretrained
        )
        self.set_new_weights(img_weight, feature_weights)

    def set_reduction(self, reduction):
        self.reduction = reduction

    def forward(self, x, y):
        if not self.mode_3d:
            return self._forward(x, y)
        else:
            final_pred = None
            for axis in self.mode_3d_apply_along_axes:
                pred = self._forward(x, y, axis)
                if final_pred is None:
                    final_pred = pred
                else:
                    final_pred += pred
            return final_pred / len(self.mode_3d_apply_along_axes)

    def _forward(self, x, y, axis=None):
        if self.reduction == 'pixelwise':
            assert (len(self.feature_weights) + (self.img_weight != 0)) == 1

        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        loss = None
        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        # preprocess
        if not self.mode_3d:
            x = self._preprocess(x)
            y = self._preprocess(y)

            f_x = self.model(x, layers)
            f_y = self.model(y, layers)
        else:
            if axis == 0:
                pass
            elif axis == 1:
                x = x.permute(0, 1, 3, 2, 4)
                y = y.permute(0, 1, 3, 2, 4)
            elif axis == 2:
                x = x.permute(0, 1, 4, 3, 2)
                y = y.permute(0, 1, 4, 3, 2)
            else:
                raise NotImplementedError()

            x_flat = x.reshape(-1, 1, *x.shape[3:])
            y_flat = y.reshape(-1, 1, *y.shape[3:])

            x_flat = self._preprocess(x_flat)
            y_flat = self._preprocess(y_flat)

            f_x = self.model(x_flat, layers)
            f_y = self.model(y_flat, layers)

            def convert(f_x, x_shape):
                f_x = [data.reshape((x_shape[0], x_shape[2], *data.shape[1:])) for data in f_x]
                f_x = [z.permute(0, 2, 1, 3, 4) for z in f_x]
                return f_x

            x_shape = x.shape
            f_x = convert(f_x, x_shape)
            f_y = convert(f_y, x_shape)

        # compute loss
        for i in range(len(f_x)):
            cur_loss = self._loss(f_x[i], f_y[i])

            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'pixelwise':
            loss = loss.unsqueeze(1)

            if self.mode_3d:
                mode='trilinear'
            else:
                mode = 'bilinear'
            loss = F.interpolate(loss, mode=mode, size=x.shape[2:])

            if axis == 0:
                pass
            elif axis == 1:
                loss = loss.permute(0, 1, 3, 2, 4)
            elif axis == 2:
                loss = loss.permute(0, 1, 4, 3, 2)
            else:
                raise NotImplementedError()

            return loss
        else:
            raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def set_new_weights(self, img_weight=0, feature_weights=None):
        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        if self.normalize_to_vgg_input:
            x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda x: torch.abs(x)
        else:
            norm = lambda x: x * x

        diff = (x - y)
        if not self.use_relative_error:
            loss = norm(diff)
        else:
            if self.mode_3d:
                means = norm(x).mean(4).mean(3).mean(2).mean(1)
                means = means.detach()
                loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1, 1))
            else:
                means = norm(x).mean(3).mean(2).mean(1)
                means = means.detach()
                loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))

        # perform reduction
        if self.reduction == 'pixelwise':
            return loss.mean(1)
        else:
            if self.mode_3d:
                return loss.mean(4).mean(3).mean(2).mean(1)
            else:
                return loss.mean(3).mean(2).mean(1)


class ReconstructionLossType(Enum):
    perceptual = 'perceptual'
    relative_perceptual_L1 = 'relative_perceptual_L1'
    l1 = 'l1'
    l2 = 'l2'


RECONSTRUCTION_LOSSES = {
    ReconstructionLossType.perceptual: PerceptualLoss,
    ReconstructionLossType.l1: L1Loss,
    ReconstructionLossType.l2: L2Loss
}