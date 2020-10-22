import os

import nibabel as nib
import numpy as np
import torch
from torchvision import transforms
import PIL
import argparse
import tqdm
import monai

from mood.dpa.train import DeepPerceptualAutoencoder
from mood.dpa.rec_losses import ReconstructionLossType
from mood.dpa.pg_rec_losses import PG_RECONSTRUCTION_LOSSES
from mood.utils.utils import load_yaml


def predict(input_folder, target_folder, config_path, mode):
    config = load_yaml(config_path)
    do_not_process_small_area = config['do_not_process_small_area']
    delete_zero_area = config['delete_zero_area']
    intermediate_resize = config['resize']

    path_to_vgg19_weights = config['path_to_vgg19_weights']

    models = []
    for params in config['models']:
        checkpoint = torch.load(params['checkpoint_path'])
        along_axis = params['along_axis']
        resize = params['resize']

        # checkpoint['config']['image_rec_loss']['loss_kwargs']['mode_3d'] = True
        checkpoint['config']['image_rec_loss']['loss_kwargs']['path_to_vgg19_weights'] = path_to_vgg19_weights
        checkpoint['config']['image_rec_loss']['loss_type'] = 'relative_perceptual_L1'
        checkpoint['config']['image_rec_loss']['loss_kwargs']['normalize_to_vgg_input'] = \
            checkpoint['config']['image_rec_loss']['loss_kwargs']['normalize_vgg_input']
        del checkpoint['config']['image_rec_loss']['loss_kwargs']['normalize_vgg_input']

        model = DeepPerceptualAutoencoder.load_anomaly_detection_model(checkpoint)
        enc, dec, image_rec_loss, (stage, resolution, _, _, _) = model

        loss_type = checkpoint['config']['image_rec_loss']['loss_type']
        loss_kwargs = checkpoint['config']['image_rec_loss']['loss_kwargs']
        loss_kwargs['mode_3d'] = True
        image_rec_loss = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](
            max_resolution=resolution, **loss_kwargs)
        image_rec_loss.set_stage_resolution(stage, resolution)
        image_rec_loss.cuda()

        model = enc, dec, image_rec_loss, _

        models.append((model, along_axis, resize))

    for filename in tqdm.tqdm(sorted(os.listdir(input_folder))):
        source_file = os.path.join(input_folder, filename)

        nimg = nib.load(source_file)
        image_3d = nimg.get_fdata()

        if intermediate_resize is None:
            sum_pred_3d = np.zeros_like(image_3d)
            count_pred_3d = np.zeros_like(image_3d)
        else:
            sum_pred_3d = np.zeros((intermediate_resize, intermediate_resize, intermediate_resize))
            count_pred_3d = np.zeros((intermediate_resize, intermediate_resize, intermediate_resize))

        for model, along_axis, resize in models:
            image_transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            tr_image = []

            zeros_before = 0
            zeros_after = 0
            find_not_zeros = False
            for i in range(image_3d.shape[along_axis]):
                slc = [slice(None)] * len(image_3d.shape)
                slc[along_axis] = i

                if do_not_process_small_area:
                    pil_image = image_3d[slc].astype(np.float32)
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
                    pil_image = image_3d[slc].astype(np.float32)
                    pil_image = PIL.Image.fromarray(pil_image, mode='F')
                    tr_image.append(image_transform(pil_image))

            tr_image = torch.stack(tr_image, dim=0).cuda()

            with torch.no_grad():
                enc, dec, image_rec_loss, _ = model
                MAX_BATCH = 128
                # MAX_BATCH = 512

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
                # print(rec_image.shape)

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

                if intermediate_resize is None:
                    pred = torch.nn.functional.interpolate(pred, mode='trilinear', size=image_3d.shape)
                else:
                    pred = torch.nn.functional.interpolate(pred, mode='trilinear',
                                                           size=(intermediate_resize, intermediate_resize, intermediate_resize))

                pred = pred.squeeze(0).squeeze(0)
                pred = pred.detach().numpy()

                if delete_zero_area:
                    pred = pred * (image_3d > 0)

            sum_pred_3d += pred
            count_pred_3d += (pred > 0)

        pred_3d = sum_pred_3d / (count_pred_3d + (count_pred_3d == 0)) / 100

        if mode == 'sample':
            anomaly_score = pred_3d.max()

            with open(os.path.join(target_folder, filename + ".txt"), "w") as write_file:
                write_file.write(str(anomaly_score))

        elif mode == 'pixel':
            if intermediate_resize is not None:
                transform = monai.transforms.Resize(image_3d.shape, mode='trilinear')
                pred_3d = transform(pred_3d[None]).squeeze(0)

            # transform = monai.transforms.Resize((128, 128, 128), mode='trilinear')
            # pred_3d = transform(pred_3d[None]).squeeze(0)

            final_nimg = nib.Nifti1Image(pred_3d.astype(np.float32), affine=nimg.affine)
            nib.save(final_nimg, os.path.join(target_folder, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("-m", "--mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False,
                        choices=['pixel', 'sample'])

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    config = args.config
    mode = args.mode

    os.makedirs(output_dir, exist_ok=True)

    predict(input_dir, output_dir, config, mode)
