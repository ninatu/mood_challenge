import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import monai


def nifti_to_2d_slices(input_folder: str, output_folder: str, axis: int, filtered, resize):
    for fname in tqdm(sorted(os.listdir(input_folder))):

        if not fname.endswith("nii.gz"):
            continue

        n_file = os.path.join(input_folder, fname)
        nifti = nib.load(n_file)

        np_data = nifti.get_fdata()
        np_data = np_data.astype(np.float16)

        f_basename = fname.split(".")[0]

        for i in range(np_data.shape[axis]):
            slc = [slice(None)] * len(np_data.shape)
            slc[axis] = i
            image = np_data[slc]

            if resize:
                tr = monai.transforms.Resize((resize, resize))
                image = tr(image[None])[0]

            if filtered:
                brain_mask = image > 0
                if brain_mask.sum() < 4000:
                    continue

            np.save(os.path.join(output_folder, f"{f_basename}_{i}.npy"), image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-a", "--axis", required=True, type=int)
    parser.add_argument("-f", "--filter", action='store_true', default=False,
                        help='Do not save slices where # of  non zero pixels < 4000')
    parser.add_argument("-r", "--resize", required=False, type=int, default=None,
                        help='Resize image while saving')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    axis = args.axis
    filtered = args.filter
    resize = args.resize
    os.makedirs(output_dir, exist_ok=True)

    nifti_to_2d_slices(input_dir, output_dir, axis, filtered, resize)
