import numpy as np
import torch
from torch import nn, nn as nn


def get_norm_layer(type, **kwargs):
    mode_3d = kwargs.get('mode_3d', False)
    if mode_3d == False:
        if type == 'none':
            return []
        elif type == 'bn':
            return [nn.BatchNorm2d(kwargs['num_features'])]
        elif type == 'in':
            return [nn.InstanceNorm2d(kwargs['num_features'])]
        else:
            raise NotImplementedError("Unknown type: {}".format(type))
    else:
        if type == 'none':
            return []
        elif type == 'bn':
            return [nn.BatchNorm3d(kwargs['num_features'])]
        elif type == 'in':
            return [nn.InstanceNorm3d(kwargs['num_features'])]
        else:
            raise NotImplementedError("Unknown type: {}".format(type))


def get_act_layer(type, **kwargs):
    if type == 'relu':
        return [nn.ReLU()]
    elif type == 'leaky_relu':
        return [nn.LeakyReLU(kwargs.get('negative_slope', 0.2), inplace=False)]
    elif type == 'tanh':
        return [nn.Tanh()]
    elif type == 'sigmoid':
        return [nn.Sigmoid()]
    elif type == 'linear':
        return []
    else:
        raise NotImplementedError("Unknown type: {}".format(type))


def get_pool_layer(type, **kwargs):
    mode_3d = kwargs.get('mode_3d', False)
    if mode_3d == False:
        if type == 'avg':
            return [nn.AvgPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
        elif type == 'max':
            return [nn.MaxPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
        else:
            raise NotImplementedError("Unknown type: {}".format(type))
    else:
        if type == 'avg':
            return [nn.AvgPool3d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
        elif type == 'max':
            return [nn.MaxPool3d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
        else:
            raise NotImplementedError("Unknown type: {}".format(type))


class Noise(nn.Module):
    def __init__(self, incoming):
        super(Noise, self).__init__()
        assert isinstance(incoming, nn.Conv2d)

        self.scale = nn.Parameter(torch.zeros(incoming.out_channels), requires_grad=True)

        self.bias = None
        if incoming.bias is not None:
            self.bias = incoming.bias
            incoming.bias = None

    def forward(self, x):
        noise = x.new_tensor(torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]))
        x = x + noise * self.scale.reshape(1, -1, 1, 1)
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type='zero', norm='none',
                 act='linear', mode_3d=False):
        super(ConvBlock, self).__init__()
        leaky_relu_param = 0.2
        layers = []

        if pad_type == 'reflect':
            layers.append(nn.ReflectionPad2d(padding))
            padding = 0
        elif pad_type == 'zero':
            pass
        else:
            raise NotImplementedError

        conv_func = nn.Conv2d if mode_3d == False else nn.Conv3d
        conv = conv_func(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain(act, param=leaky_relu_param))
        layers.append(conv)

        layers += get_norm_layer(norm, num_features=out_channels, mode_3d=mode_3d)
        layers += get_act_layer(act, negative_slope=leaky_relu_param)

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', lambd=1, mode_3d=False):
        super(ResBlock, self).__init__()
        self.lambd = lambd

        model = []
        model += [ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type,mode_3d=mode_3d)]
        model += [ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type, mode_3d=mode_3d)]
        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear',
                                     pad_type=pad_type, mode_3d=mode_3d)

    def forward(self, x):
        return self.skipcon(x) + self.lambd * self.model(x)


class PreActResnetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', lambd=1, mode_3d=False):
        super(PreActResnetBlock, self).__init__()
        self.lambd = lambd

        model = []
        model += get_norm_layer(norm, num_features=input_dim, mode_3d=mode_3d)
        model += get_act_layer(act)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type, mode_3d=mode_3d),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type, mode_3d=mode_3d)
        ]
        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear',
                                     pad_type=pad_type, mode_3d=mode_3d)

    def forward(self, x):
        return self.skipcon(x) + self.lambd * self.model(x)


class PreActResnetBlockUp(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', upsample_mode='nearest', mode_3d=False):
        super(PreActResnetBlockUp, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim, mode_3d=mode_3d)
        model += get_act_layer(act)
        model += [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type, mode_3d=mode_3d),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type, mode_3d=mode_3d)
        ]
        self.model = nn.Sequential(*model)

        skipcon = [nn.Upsample(scale_factor=2, mode='nearest')]
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear',
                                  pad_type=pad_type, mode_3d=mode_3d)]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class PreActResnetBlockDown(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', pool='avg', mode_3d=False):
        super(PreActResnetBlockDown, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim, mode_3d=mode_3d)
        model += get_act_layer(act)
        model += get_pool_layer(pool, kernel_size=2, stride=2, mode_3d=mode_3d)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type, mode_3d=mode_3d),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type, mode_3d=mode_3d),
        ]
        self.model = nn.Sequential(*model)

        skipcon = get_pool_layer(pool, kernel_size=2, stride=2, mode_3d=mode_3d)
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear',
                                  pad_type=pad_type, mode_3d=mode_3d)]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class EqualLayer(nn.Module):
    def forward(self, x):
        return x


class ConcatLayer(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatLayer, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        return y


class FadeinLayer(nn.Module):
    def __init__(self, ):
        super(FadeinLayer, self).__init__()
        self._alpha = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

    def set_progress(self, alpha):
        self._alpha.data[0] = np.clip(alpha, 0, 1.0)

    def get_progress(self):
        return self._alpha.data.cpu().item()

    def forward(self, x):
        return torch.add(x[0].mul(1.0 - self._alpha), x[1].mul(self._alpha))

    def __repr__(self):
        return self.__class__.__name__ + '(get_alpha = {:.2f})'.format(self._alpha.data[0])