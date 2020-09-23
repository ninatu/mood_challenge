import yaml
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import itertools
import monai


def update_dict(obj, new_values):
    if obj is None:
        return new_values
    for k, v in new_values.items():
        if isinstance(v, dict) and k in obj.keys():
            obj[k] = update_dict(obj[k], new_values[k])
        elif obj is None:
            obj = {k: new_values[k]}
        else:
            obj[k] = new_values[k]
    return obj


def load_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)


def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.dump(data, open(path, 'w'))


def update_config(config, overwrite_params):
    overwrite_params = yaml.load(overwrite_params, Loader=yaml.FullLoader)
    return update_dict(config, overwrite_params)


def draw_image(image):
    return PIL.Image.fromarray((np.array(image) * 255).astype(np.uint8))


def read_nii_file(path):
    nifti = nib.load(path)
    data_array = nifti.get_data()
    affine_matrix = nifti.affine
    return data_array


def create_pltfig_from_path(path, n=5, axis=0):
    nifti = nib.load(path)
    image = nifti.get_data()
    return create_pltfig_of_slices(image, n=n, axis=axis)


def create_pltfig_from_path_128(path, n=5, axis=0):
    nifti = nib.load(path)
    image = nifti.get_data()

    transform = monai.transforms.Resize((128, 128, 128), mode='trilinear')
    image = transform(image[None]).squeeze(0)
    image = np.clip(image, a_max=1, a_min=0)
    image = image > 0.5

    return create_pltfig_of_slices(image, n=n, axis=axis)


def create_pltfig_of_slices(image, n=5, axis=0):
    f, axs = plt.subplots(n, n, figsize=(20,20), squeeze=False)
    axs = list(itertools.chain.from_iterable(axs))

    step = int(image.shape[2] // (n * n))

    for i in range(1, n * n + 1):
        slc = [slice(None)] * len(image.shape)
        slc[axis] = i * step
        axs[i - 1].imshow(image[slc],  cmap='gray')
        axs[i - 1].axis('off')
        axs[i - 1].set_title(str(i * step))

    plt.tight_layout()
    return f
