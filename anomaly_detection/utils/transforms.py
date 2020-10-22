from torchvision import transforms
import monai.transforms

from anomaly_detection.utils.datasets import DatasetType


class Transform2D:
    def __init__(self, resize=None, to_tensor=True, normalize=True):
        tr = []

        if resize is not None:
            tr += [transforms.Resize((resize, resize))]

        if to_tensor:
            tr += [transforms.ToTensor()]

        if normalize:
            tr += [transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(tr)

    def __call__(self, img):
        return self.transform(img)


class Transform3D:
    def __init__(self, resize=None, normalize=True):
        tr = []

        if resize is not None:
            tr += [monai.transforms.Resize((resize, resize, resize))]

        tr += [monai.transforms.ToTensor()]

        if normalize:
            tr += [Normalize3D(0.5, 0.5)]
        self.transform = transforms.Compose(tr)

    def __call__(self, img):
        return self.transform(img[None])


TRANSFORMS = {
    DatasetType.numpy2d: Transform2D,
    DatasetType.nifti3d: Transform3D,
}


class Normalize3D(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return tensor.sub_(self.mean).div_(self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

