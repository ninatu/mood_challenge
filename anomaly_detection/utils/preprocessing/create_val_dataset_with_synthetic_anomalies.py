import os
import numpy as np
from tqdm import tqdm
import argparse
import nibabel as nib
import pyellipsoid.drawing
import pandas as pd


def get_center_and_radius(mask):
    def get_center(row):
        return np.array(range(len(row)))[row > 0].sum() / (row > 0).sum()

    row = mask.sum(axis=0)
    h = get_center(row)
    h_r = (row > 0).sum() / 2

    row = mask.sum(axis=1)
    w = get_center(row)
    w_r = (row > 0).sum() / 2

    return h, w, (h_r + w_r) / 2


def gen_coord(h, r):
    return np.random.normal(h, r / 3)


def gen_radius(r):
    return max(5, np.random.normal(r / 16, r / 16))


def gen_color():
    return np.random.rand()


def gen_noise(color, size):
    return np.random.rand(size) * color / 10


def gen_alpha():
    return np.clip(np.random.exponential(scale=1 / 4), 0, 1)


def get_3d_center_and_radius(mask):
    def get_center(row):
        return np.array(range(len(row)))[row > 0].sum() / (row > 0).sum()
    def get_radius(row):
        return (row > 0).sum() / 2
    row = mask.sum(axis=2).sum(axis=1)
    h, h_r = get_center(row), get_radius(row)

    row = mask.sum(axis=2).sum(axis=0)
    w, w_r = get_center(row), get_radius(row)

    row = mask.sum(axis=1).sum(axis=0)
    c, c_r = get_center(row), get_radius(row)

    return h, w, c, (h_r + w_r + c_r) / 3


def gen_angle():
    return np.deg2rad(np.random.uniform(0, 90))


def draw_ellipsoid(image, mask, h, w, c, r):
    centers = gen_coord(h, r), gen_coord(w, r), gen_coord(c, r)
    radius = gen_radius(r), gen_radius(r), gen_radius(r)
    angles = gen_angle(), gen_angle(), gen_angle()

    c = gen_color()
    alpha = gen_alpha()
    ellipse = pyellipsoid.drawing.make_ellipsoid_image(image.shape, centers, radius, angles) > 0
    image_slice = image[ellipse]
    image[ellipse] = alpha * image_slice + (1 - alpha) * (c + gen_noise(c, len(image_slice)))
    np.clip(image, 0, 1, out=image)
    mask[ellipse] = 1


def create_anomaly_3d_dataset(input_folder, output_image_folder, output_mask_folder, folds_path=None, fold=0):
    zero_mask_path = None

    EXT = ".nii.gz"
    filenames = os.listdir(input_folder)
    filenames = [name for name in filenames if name.endswith(EXT)]

    if folds_path is not None:
        folds = pd.read_csv(folds_path)
        folds = folds[folds.test_fold == int(fold)]
        fold_filenames = folds.filename
        fold_filenames = [name.replace('.nii.gz', '') for name in fold_filenames]

        filtered_filenames = []
        for name in filenames:
            basename = name.replace(EXT, '')
            basename = basename.split('_')[0]
            if basename in fold_filenames:
                filtered_filenames.append(name)
        filenames = filtered_filenames

    for fname in tqdm(sorted(filenames)):
        if not fname.endswith(EXT):
            continue
        base_name = fname.replace(EXT, '')

        path = os.path.join(input_folder, fname)
        nimage = nib.load(path)
        image = nimage.get_fdata()
        affine = nimage.affine
        image = image.astype(np.float32)
        mask = np.zeros_like(image).astype(np.uint8)

        # Add normal image
        os.link(path, os.path.join(output_image_folder, fname))
        if zero_mask_path is None:
            zero_mask_path = os.path.join(output_mask_folder, fname)
            nib.save(nib.Nifti1Image(mask, affine=affine), zero_mask_path)
        else:
            os.link(zero_mask_path, os.path.join(output_mask_folder, fname))

        # Create and add abnormal image
        brain_mask = image > 0
        h, w, c, r = get_3d_center_and_radius(brain_mask)
        for _ in range(np.random.randint(1, 4)):
            draw_ellipsoid(image, mask, h, w, c, r)

        outname = f'{base_name}_anom{EXT}'
        nib.save(nib.Nifti1Image(image, affine=affine), os.path.join(output_image_folder, outname))
        nib.save(nib.Nifti1Image(mask, affine=affine), os.path.join(output_mask_folder, outname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dir",
                        default='./data/original/brain_train/',
                        help='input_dir')
    parser.add_argument("-o", "--output_image_dir",
                        default='./data/preprocessed/brain_train/3d_test',
                        help='output_image_dir')
    parser.add_argument("-m", "--output_mask_dir",
                        default='./data/preprocessed/brain_train/3d_test_masks/',
                        help='output_mask_dir')
    parser.add_argument("-p", "--folds_path", required=False, type=str, default=None,
                        help='Path to csv file with folds info. '
                             'Use if you want to create a synthetic dataset only from one "test" fold of input dataset')
    parser.add_argument("-f", "--fold", required=False, type=str, default=None,
                        help='# of fold. '
                             'Use if you want to create a synthetic dataset only from one "test" fold of input dataset')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_image_dir = args.output_image_dir
    output_mask_dir = args.output_mask_dir
    folds_path = args.folds_path
    fold = args.fold

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    create_anomaly_3d_dataset(input_dir, output_image_dir, output_mask_dir, folds_path, fold)
