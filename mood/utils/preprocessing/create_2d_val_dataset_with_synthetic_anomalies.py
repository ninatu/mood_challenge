import os
import numpy as np
from tqdm import tqdm
import argparse
import skimage.draw
import pandas as pd

from mood.utils.preprocessing.create_val_dataset_with_synthetic_anomalies \
    import gen_coord, gen_radius, gen_alpha, get_center_and_radius, gen_color, gen_noise


def draw_circle(image, mask, h, w, r):
    x_h, x_w = gen_coord(h, r), gen_coord(w, r)
    radius = gen_radius(r)

    c = gen_color()
    alpha = gen_alpha()

    rr, cc = skimage.draw.circle(x_w, x_h, radius, image.shape)
    image[rr, cc] = alpha * image[rr, cc] + (1 - alpha) * (c + gen_noise(c, len(rr)))
    np.clip(image, 0, 1, out=image)
    mask[rr, cc] = 1


def draw_ellipse(image, mask, h, w, r):
    x_h, x_w = gen_coord(h, r), gen_coord(w, r)
    radius1 = gen_radius(r)
    radius2 = gen_radius(r)

    c = gen_color()
    alpha = gen_alpha()
    rr, cc = skimage.draw.ellipse(x_w, x_h, radius1, radius2, image.shape)
    image[rr, cc] = alpha * image[rr, cc] + (1 - alpha) * (c + gen_noise(c, len(rr)))
    np.clip(image, 0, 1, out=image)
    mask[rr, cc] = 1


def create_anomaly_2d_dataset(input_folder, output_image_folder, output_mask_folder, folds_path, fold):
    zero_mask_path = None

    filename_endwith = '.npy'
    filenames = os.listdir(input_folder)
    # print(len(filenames))
    filenames = [name for name in filenames if name.endswith(filename_endwith)]
    # print(len(filenames))

    if folds_path is not None:
        folds = pd.read_csv(folds_path)
        folds = folds[folds.test_fold == int(fold)]
        fold_filenames = folds.filename
        fold_filenames = [name.replace('.nii.gz', '') for name in fold_filenames]
        # print(len(fold_filenames), fold_filenames[:10])

        filtered_filenames = []
        for name in filenames:
            basename = name.replace(filename_endwith, '')
            basename = basename.split('_')[0]
            # print(basename)
            if basename in fold_filenames:
                filtered_filenames.append(name)
        filenames = filtered_filenames

    for fname in tqdm(sorted(filenames)):
        base_name, ext = os.path.splitext(fname)
        path = os.path.join(input_folder, fname)
        image = np.load(path)
        image = np.array(image).astype(np.float32)
        mask = np.zeros_like(image).astype(np.uint8)

        # Add normal image
        os.link(path, os.path.join(output_image_folder, fname))
        if zero_mask_path is None:
            zero_mask_path = os.path.join(output_mask_folder, fname)
            np.save(zero_mask_path, mask)
        else:
            os.link(zero_mask_path, os.path.join(output_mask_folder, fname))

        # Create and add abnormal image
        brain_mask = image > 0
        if brain_mask.sum() < 4000:
            continue

        h, w, r = get_center_and_radius(brain_mask)
        draw_func = [draw_circle, draw_ellipse][np.random.randint(0, 2)]
        for _ in range(np.random.randint(1, 4)):
            draw_func(image, mask, h, w, r)

        outname = f'{base_name}_anom{ext}'
        np.save(os.path.join(output_image_folder, outname), image.astype(np.float16))
        np.save(os.path.join(output_mask_folder, outname), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dir",
                        default='./data/preprocessed/brain_train/2d_axis_2',
                        help='input_dir')
    parser.add_argument("-o", "--output_image_dir",
                        default='./data/preprocessed/brain_train/2d_axis_2_test',
                        help='output_image_dir')
    parser.add_argument("-m", "--output_mask_dir",
                        default='./data/preprocessed/brain_train/2d_axis_2_test_masks/',
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

    create_anomaly_2d_dataset(input_dir, output_image_dir, output_mask_dir, folds_path, fold)
