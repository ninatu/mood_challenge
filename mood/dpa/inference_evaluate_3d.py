import nibabel as nib

import argparse
import yaml
import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import PIL.Image
import monai

from mood.dpa.evaluate import save_sample_score
from mood.dpa.train import DeepPerceptualAutoencoder
from mood.utils.datasets import DatasetType, DATASETS
from mood.utils.transforms import TRANSFORMS
from mood.dpa.rec_losses import ReconstructionLossType
from mood.dpa.pg_rec_losses import PG_RECONSTRUCTION_LOSSES


def main(config):
    verbose = config['verbose']
    results_root = config['results_root']
    model_path = config['test_model_path']
    save_inference = config['save_inference']

    along_axis = config['apply_along_axis']
    test_image_rec_loss = config.get('test_image_rec_loss')
    score_reduction = config['score_reduction']
    delete_zero_area = config['delete_zero_area']
    do_not_process_small_area = config['do_not_process_small_area']
    resize_3d_for_evaluation = config.get('resize_3d_for_evaluation')

    os.makedirs(results_root, exist_ok=True)

    if verbose:
        print(yaml.dump(config, default_flow_style=False))

    enc, dec, image_rec_loss, (stage, resolution, progress, n_iter, mix_res_module) = \
        DeepPerceptualAutoencoder.load_anomaly_detection_model(torch.load(model_path))

    if test_image_rec_loss is not None:
        loss_type = config['test_image_rec_loss']['loss_type']
        loss_kwargs = config['test_image_rec_loss']['loss_kwargs']
        image_rec_loss = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](
            max_resolution=resolution, **loss_kwargs)
        image_rec_loss.set_stage_resolution(stage, resolution)
        image_rec_loss.cuda()

    dataset_type = DatasetType(config['test_dataset']['dataset_type'])
    dataset_kwargs = config['test_dataset']['dataset_kwargs']
    transform_kwargs = config['test_dataset']['transform_kwargs']

    image_transform = TRANSFORMS[DatasetType.numpy2d](**transform_kwargs)

    dataset = DATASETS[dataset_type](
        return_image_name=True,
        **dataset_kwargs
    )

    results_root = config['results_root']

    sample_true = []
    pixel_true = []

    sample_pred = []
    pixel_pred = []

    sample_output_dir = os.path.join(results_root, 'sample')
    pixel_output_dir = os.path.join(results_root, 'pixel')

    for image, mask, affine, name in tqdm.tqdm(dataset):
        tr_image = []

        zeros_before = 0
        zeros_after = 0
        find_not_zeros = False
        for i in range(image.shape[along_axis]):
            slc = [slice(None)] * len(image.shape)
            slc[along_axis] = i

            if do_not_process_small_area:
                pil_image = image[slc].astype(np.float32)
                if pil_image.sum() > 4000:
                    find_not_zeros = True
                    pil_image = PIL.Image.fromarray(pil_image, mode='F')
                    tr_image.append(image_transform(pil_image))
                else:
                    if find_not_zeros:
                        zeros_after += 1
                    else:
                        zeros_before += 1
            else:
                pil_image = image[slc].astype(np.float32)
                pil_image = PIL.Image.fromarray(pil_image, mode='F')
                tr_image.append(image_transform(pil_image))

        tr_image = torch.stack(tr_image, dim=0).cuda()

        with torch.no_grad():
            # TODO: FIX IT. this hack is for testing large 3d scans (in order to fit in 12 Gb GPU memory)
            MAX_BATCH = 128
            if len(tr_image) < MAX_BATCH:
                rec_image = dec(enc(tr_image)).detach()
            else:
                rec_image = None
                for i in range(0, len(tr_image), MAX_BATCH):
                    y = dec(enc(tr_image[i: i + MAX_BATCH])).detach()
                    if rec_image is None:
                        rec_image = y
                    else:
                        rec_image = torch.cat((rec_image, y), dim=0)
            # END todo. simple solution: rec_image = dec(enc(tr_image)).detach()

            tr_image = tr_image.squeeze(1).unsqueeze(0).unsqueeze(0)
            rec_image = rec_image.squeeze(1).unsqueeze(0).unsqueeze(0)

            image_rec_loss.set_reduction('pixelwise')
            pred = image_rec_loss(tr_image, rec_image)
            pred = pred.squeeze(0).squeeze(0).unsqueeze(1)

            assert pred.size(1) == 1
            pred = pred[:, 0]

            if do_not_process_small_area:
                slice_shape = pred.shape[1:]
                pred = torch.cat((
                    torch.zeros((zeros_before, *slice_shape)).cuda(),
                    pred,
                    torch.zeros((zeros_after, *slice_shape)).cuda()
                ))

            if along_axis == 0:
                pass
            elif along_axis == 1:
                pred = pred.permute(1, 0, 2)
            elif along_axis == 2:
                pred = pred.permute(1, 2, 0)
            else:
                raise NotImplementedError()
            pred = pred.unsqueeze(0).unsqueeze(0).detach().cpu()

            pred = torch.nn.functional.interpolate(pred, mode='trilinear', size=image.shape)
            pred = pred.squeeze(0).squeeze(0)
            pred = pred.detach().numpy()

            if delete_zero_area:
                pred = pred * (image > 0)

        if save_inference:
            save_3d_pixel_score(pixel_output_dir, pred, affine, name)

        if score_reduction == 'mean':
            anomaly_score = pred.mean()
        elif score_reduction == 'max':
            anomaly_score = pred.max()
        else:
            raise NotImplementedError()
        save_sample_score(sample_output_dir, [anomaly_score], [name])

        if resize_3d_for_evaluation is not None:
            transform = monai.transforms.Resize(
                (resize_3d_for_evaluation, resize_3d_for_evaluation, resize_3d_for_evaluation),
                mode='trilinear')
            mask = transform(mask[None]).squeeze(0)
            pred = transform(pred[None]).squeeze(0)

        sample_true.append(mask.sum() > 0)
        sample_pred.append(anomaly_score)
        pixel_true.append(mask.flatten() > 0.5)
        pixel_pred.append(pred.flatten())

    sample_true = np.array(sample_true).astype(np.bool)
    sample_pred = np.array(sample_pred)
    pixel_true = np.concatenate(pixel_true).astype(np.bool)
    pixel_pred = np.concatenate(pixel_pred)

    ap_sample = average_precision_score(sample_true, sample_pred)
    print("AP sample", ap_sample)
    auc_sample = roc_auc_score(sample_true, sample_pred)
    print("AUC sample", auc_sample)

    ap_pixel = average_precision_score(pixel_true, pixel_pred)
    print("AP pixel", ap_pixel)
    auc_pixel = roc_auc_score(pixel_true, pixel_pred)
    print("AUC pixel", auc_pixel)

    scores = pd.DataFrame(
        [[ap_sample, auc_sample],
         [ap_pixel, auc_pixel]],
        columns=['AP', 'ROC AUC'], index=['sample', 'pixel']
    )

    print(scores)

    scores.to_csv(os.path.join(results_root, 'scores.csv'))


def save_3d_pixel_score(output_dir, score, affine, image_name):
    os.makedirs(output_dir, exist_ok=True)
    final_nimg = nib.Nifti1Image(score, affine=affine)
    nib.save(final_nimg, os.path.join(output_dir, image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str, nargs='*', help='Path to eval config')

    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        main(config)
