# Medical Out-of-Distribution Analysis Challenge. Solution. Third Place in Pixel-Level Task


[![License][license-shield]][license-url]


## Challenge

[Offical Challenge Website](http://medicalood.dkfz.de/web/)

[Github-Repository](https://github.com/MIC-DKFZ/mood)

Medical Out-of-Distribution Analysis Challenge (MOOD) presented two tasks (sample-level 
and pixel-level out-of-distribution detection) on two medical datasets 
(brain MRI-dataset and abdominal CT-dataset).

## Solution 

Presentation of the solution (starts at 18:04 minutes): [https://www.youtube.com/watch?v=yOemj1TQfZU](https://www.youtube.com/watch?v=yOemj1TQfZU)

We based our solution on [Deep Perceptual Autoencoder.](https://arxiv.org/pdf/2006.13265.pdf) 
We applied a Deep Perceptual Autoencoder on 2D slices of 3D volumes. 
To calculate [Perceptual Loss](https://arxiv.org/abs/1603.08155), we used VGG19 network
as feature extractor pre-trained
using an unsupervised learning framework [SimCLR](https://arxiv.org/abs/2002.05709).

Our training procedure consist of two stages: 
1. SimCLR training of VGG19 features on joined set of all sliced of 3D volume along 0'th, 1'st, 2'nd axes. 
2. Training three Deep Perceptual Autoencoders -- each on the set of 2D slices of 3D volume along for the corresponded axes.

We used Deep Perceptual Autoencoders to predict anomalies pixel-wise (giving an abnormality score for each voxel of 3D volume), 
and sample-wise (for a whole 3D volume):
1. Pixel-wise abnormality scores. The final pixel-wise prediction was the average of pixel-wise predictions over three models (applied along different axes). To obtain pixel-level prediction, we change the computation of the L1-norm over a whole feature map to the pixel-wise L1-norm in the numerator of Equation~\ref{eq:loss}. After obtaining such a map of reconstruction errors, we resized this map to an input image shape. 
2. Sample-wise abnormality score. As an abnormality score of a whole 3D volume, we used a maximum of volume-level abnormality scores.  


## Structure of Project 
    anomaly_detection - Python Package. Implementation of Deep Perceptual Autoencoder (https://arxiv.org/pdf/2006.13265.pdf).
                        Installation: 
                            pip install -r requirements.txt
                            pip install -e . --user
    configs - Yaml Configs used in our solution
        └───abdom - configs used to train final models on abdominal CT-dataset
        │   │───pixel
        |   |   │───axis_0
        |   |   │   |   train_config.yaml
        |   |   │───axis_2
        |   |   │   |   train_config.yaml
        │   │───sample
        |   |   │ ...
        └───brain - configs used to train final models on  brain MRI-dataset
        |   |   ....
        └───cross_validation - configs used to search hyperparameters
        │   │───axis_0
        │   │    meta_train.yaml
        │   │    meta_inference_3d.yaml
        │   │───cv
        |   |   │───res_128_lr_1_ld_64_pf_r32
        |   |   |   │───0
        |   |   |   |   train_config.yaml -- training config
        |   |   |   |   inference_3d_config -- evaluation config
        |   |   | ...
     submission_data -- Scripts and configs of the final model
     folds -- Folds used in cross-validation

## Installation 

```bash
pip install -r requirements.txt
pip install -e . --user
```
     
## Data Preparation

1. Download data (see [Challenge Website](http://medicalood.dkfz.de/web/))
2. Save 2D slices along all axes
    ```bash
    python anomaly_detection/utils/data/save_2D.py [-h] -i INPUT_DIR -o OUTPUT_DIR -a AXIS [-f] [-r RESIZE]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT_DIR, --input_dir INPUT_DIR
      -o OUTPUT_DIR, --output_dir OUTPUT_DIR
      -a AXIS, --axis AXIS
      -f, --filter          Do not save slices where # of non zero pixels < 4000
      -r RESIZE, --resize RESIZE
                            Resize image while saving
   ```
3. Create folds for cross-validation or use ours (`folds` dir)
    ```bash
    python anomaly_detection/utils/data/create_folds.py [-h] -i INPUT_DIR -o OUTPUT_PATH [-n N_FOLDS]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT_DIR, --input_dir INPUT_DIR
      -o OUTPUT_PATH, --output_path OUTPUT_PATH
      -n N_FOLDS, --n_folds N_FOLD
   ```
4. Optionally: create a synthetic dataset for validation
    ```bash
    python anomaly_detection/utils/data/create_val_dataset_with_synthetic_anomalies.py [-h] -i INPUT_DIR -o OUTPUT_IMAGE_DIR -m OUTPUT_MASK_DIR [-p FOLDS_PATH] [-f FOLD]

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT_DIR, --input_dir INPUT_DIR
      -o OUTPUT_IMAGE_DIR, --output_image_dir OUTPUT_IMAGE_DIR
      -m OUTPUT_MASK_DIR, --output_mask_dir OUTPUT_MASK_DIR
      -p FOLDS_PATH, --folds_path FOLDS_PATH
                            Path to csv file with folds info. Use if you want to create a synthetic dataset only from one "test" fold of input dataset
      -f FOLD, --fold FOLD  # of fold. Use if you want to create a synthetic dataset only from one "test" fold of input dataset
   ```

## Training

### Pre-training of VGG19 features
    
Since no other data and data sources were allowed to use in the challenge,
we used an unsupervised learning framework [SimCLR](https://arxiv.org/abs/2002.05709)
to pre-train VGG19 features (used in the perceptual loss)

See [our fork](https://github.com/ninatu/SimCLR) of implementation 
of SimCLR adapted for the VGG19 training on provided data.
    

### Training Deep Perceptual Autoencoder

See examples of configs for training and inference in `configs` dir.

To train Deep Perceptual Autoencoder (DPA), run:
```bash
python anomaly_detection/main.py train {path_to_config}
```

To inference and evaluate your model on synthetic dataset, run
```bash
python anomaly_detection/main.py inference_evaluate_3d {path_to_config}
```

## Building Docker With Final Model

Our final prediction was the average of predictions over three models 
in brain MRI task (applied along different axes), and over two models in abdominal CT task
(applied along 0'th and 2'th axes).

In order to build a docker with the final model:
1. Put your trained model in folder `submission_data`
2. Run 
```bash
docker build . -t mood:latest
```

Inference using the docker:
```bash
docker run --gpus all -v {input_dir}:/mnt/data -v {output_dir}:/mnt/pred mood:latest sh /workspace/run_{sample/pixel}_{TASK}.sh /mnt/data /mnt/pred
```


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://github.com/ninatu/mood_challenge/blob/master/LICENSE
