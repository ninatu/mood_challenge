import argparse
import yaml
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import nibabel as nib

from anomaly_detection.dpa.train import DeepPerceptualAutoencoder
from anomaly_detection.utils.data.datasets import DatasetType, DATASETS
from anomaly_detection.utils.data.transforms import TRANSFORMS
from anomaly_detection.dpa.rec_losses import ReconstructionLossType
from anomaly_detection.dpa.pg_rec_losses import PG_RECONSTRUCTION_LOSSES


def main(config):
    verbose = config['verbose']
    batch_size = config['test_batch_size']
    results_root = config['results_root']
    model_path = config['test_model_path']
    save_inference = config['save_inference']
    test_resize = config.get('test_resize')
    test_image_rec_loss = config.get('test_image_rec_loss')
    score_reduction = config['score_reduction']

    os.makedirs(results_root, exist_ok=True)

    if verbose:
        print(yaml.dump(config, default_flow_style=False))

    loaded_model = DeepPerceptualAutoencoder.load_anomaly_detection_model(torch.load(model_path))
    if test_image_rec_loss is not None:
        enc, dec, image_rec_loss, (stage, resolution, progress, n_iter, mix_res_module) = loaded_model
        loss_type = config['test_image_rec_loss']['loss_type']
        loss_kwargs = config['test_image_rec_loss']['loss_kwargs']
        image_rec_loss = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](
            max_resolution=resolution, **loss_kwargs)
        image_rec_loss.set_stage_resolution(stage, resolution)
        image_rec_loss.cuda()
        loaded_model = enc, dec, image_rec_loss, (stage, resolution, progress, n_iter, mix_res_module)

    dataset_type = DatasetType(config['test_dataset']['dataset_type'])
    dataset_kwargs = config['test_dataset']['dataset_kwargs']
    transform_kwargs = config['test_dataset']['transform_kwargs']

    transform = TRANSFORMS[dataset_type](**transform_kwargs)
    mask_transform = TRANSFORMS[dataset_type](**transform_kwargs, normalize=False)

    dataset = DATASETS[dataset_type](
        transform=transform,
        mask_transform=mask_transform,
        return_image_name=True,
        **dataset_kwargs
    )

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    if verbose:
        data_loader = tqdm(data_loader)

    results_root = config['results_root']

    sample_true = []
    pixel_true = []

    sample_pred = []
    pixel_pred = []

    sample_output_dir = os.path.join(results_root, 'sample')
    pixel_output_dir = os.path.join(results_root, 'pixel')
    for data in data_loader:
        if len(data) == 3:
            images, masks, names = data
        elif len(data) == 4:
            images, masks, affine, names = data
        else:
            raise NotImplementedError()

        images = images.cuda()
        masks = masks.numpy()

        pixel_true.append((masks > 0).flatten())
        sample_true.append(masks.sum(axis=tuple(range(1, len(masks.shape)))) > 0)

        pred = DeepPerceptualAutoencoder.predict_anomaly_score(loaded_model, images, reduction='pixelwise')
        pred = pred.detach().cpu().numpy()
        assert pred.shape[1] == 1
        pred = pred.squeeze(1)

        pixel_pred.append(pred.flatten())
        if save_inference:
            save_pixel_score(pixel_output_dir, pred, data)

        if score_reduction == 'mean':
            anomaly_score = [x.mean() for x in pred]
        elif score_reduction == 'max':
            anomaly_score = [x.max() for x in pred]
        else:
            raise NotImplementedError()

        sample_pred.append(anomaly_score)
        save_sample_score(sample_output_dir, anomaly_score, names)

    sample_true = np.concatenate(sample_true).astype(np.bool)
    sample_pred = np.concatenate(sample_pred)
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
        [[ap_sample, ap_pixel],
         [ap_pixel, auc_pixel]],
        columns=['AP', 'ROC AUC'], index=['sample', 'pixel']
    )

    print(scores)

    scores.to_csv(os.path.join(results_root, 'scores.csv'))


def save_sample_score(output_dir, scores, image_names):
    os.makedirs(output_dir, exist_ok=True)
    for score, image_name in zip(scores, image_names):
        with open(os.path.join(output_dir, f"{image_name}.txt"), "w") as write_file:
            write_file.write(str(score))


def save_pixel_score(output_dir, scores, data):
    print(scores.shape)
    os.makedirs(output_dir, exist_ok=True)
    if len(scores.shape) == 3:
        images, masks, names = data
        for score, image_name in zip(scores, names):
            np.save(os.path.join(output_dir, image_name), score)
    elif len(scores.shape) == 4:
        images, masks, affines, names = data
        for score, affine, image_name in zip(scores, affines, names):
            print(image_name)
            img = nib.Nifti1Image(score, affine=affine)
            nib.save(img, os.path.join(output_dir, image_name))
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str, nargs='*', help='Path to eval config')

    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        main(config)
