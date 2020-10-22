import os
from torch.utils.data import Dataset
import numpy as np
from enum import Enum
import PIL.Image
import pandas as pd
import nibabel as nib
import tqdm


class NumpyDatasetBase(Dataset):
    def __init__(self, image_root, folds_path, fold, split, filename_endwith,  mask_root=None, transform=None,
                 mask_transform=None, return_image_name=False):

        self.image_root = image_root
        self.mask_root = mask_root
        self.folds_path = folds_path
        self.fold = fold
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_image_name = return_image_name

        folds = pd.read_csv(folds_path)
        if split == 'train':
            folds = folds[folds.test_fold != self.fold]
        elif split == 'val':
            folds = folds[folds.test_fold == self.fold]
        elif split == 'all':
            folds = folds
        else:
            raise NotImplementedError()

        self.filenames = folds.filename
        self.filenames = [name.replace('.nii.gz', '') for name in self.filenames]

        image_names = os.listdir(self.image_root)
        image_names = [name for name in image_names if name.endswith(filename_endwith)]
        self.image_names = []
        for name in image_names:
            basename = name.replace(filename_endwith, '')
            basename = basename.split('_')[0]
            if basename in self.filenames:
                self.image_names.append(name)

    def __len__(self):
        return len(self.image_names)


class Numpy2DDataset(NumpyDatasetBase):
    def __init__(self, image_root, folds_path, fold, split, mask_root=None, transform=None,
                 mask_transform=None, return_image_name=False, brain_size_path=None, filter_brain_size=None,
                 cache=False, max_dataset_size=None, channels3=False):
        super().__init__(image_root, folds_path, fold, split, filename_endwith='.npy',
                         mask_root=mask_root, transform=transform, mask_transform=mask_transform,
                         return_image_name=return_image_name)

        self.brain_size_path = brain_size_path
        self.filter_brain_size = filter_brain_size
        self.max_dataset_size = max_dataset_size
        self.channels3 = channels3

        if self.filter_brain_size is not None:
            assert self.brain_size_path is not None

            brain_size = pd.read_csv(self.brain_size_path)
            unknown_image_names = list(set(self.image_names).difference(brain_size.filename))

            brain_size = brain_size[brain_size.filename.isin(self.image_names)]
            filtered_image_names = list(brain_size[brain_size.brain_size >= self.filter_brain_size].filename)
            self.image_names = unknown_image_names + filtered_image_names

        if self.max_dataset_size is not None:
            np.random.shuffle(self.image_names)
            self.image_names = self.image_names[:max_dataset_size]

        self.cache_data = {}
        self.cache = False
        if cache:
            print("Caching ....")
            for i in tqdm.tqdm(range(len(self))):
                self.cache_data[i] = self[i]
            self.cache = True

    def __getitem__(self, idx):
        if self.cache:
            return self.cache_data[idx]

        image_name = self.image_names[idx]
        image = np.load(os.path.join(self.image_root, image_name))
        if not self.channels3:
            image = PIL.Image.fromarray(image.astype(np.float32), mode='F')
        else:
            image = PIL.Image.fromarray((np.array(image) * 255).astype(np.uint8))
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mask_root is None:
            if self.return_image_name:
                return image, image_name
            else:
                return image

        mask = np.load(os.path.join(self.mask_root, image_name))
        mask = PIL.Image.fromarray(mask.astype(np.float32), mode='F')

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.return_image_name:
            return image, mask, image_name
        else:
            return image, mask


class Nifti3DDataset(NumpyDatasetBase):
    def __init__(self, image_root, folds_path, fold, split, mask_root=None, transform=None,
                 mask_transform=None, return_image_name=False):
        super().__init__(image_root, folds_path, fold, split, filename_endwith='.nii.gz',
                         mask_root=mask_root, transform=transform, mask_transform=mask_transform,
                         return_image_name=return_image_name)
        self.image_names = sorted(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        nimage = nib.load(os.path.join(self.image_root, image_name))

        image = nimage.get_data().astype(np.float32)
        affine = nimage.affine

        if self.transform:
            image = self.transform(image)

        if self.mask_root is None:
            if self.return_image_name:
                return image, image_name
            else:
                return image

        nmask = nib.load(os.path.join(self.mask_root, image_name))
        mask = nmask.get_data()

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.return_image_name:
            return image, mask, affine, image_name
        else:
            return image, mask


class DatasetType(Enum):
    numpy2d = 'numpy2d'
    nifti3d = 'nifti3d'


DATASETS = {
    DatasetType.numpy2d: Numpy2DDataset,
    DatasetType.nifti3d: Nifti3DDataset,
}
