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
    mood - Python Package. Implementation of Deep Perceptual Autoencoder (https://arxiv.org/pdf/2006.13265.pdf).
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

1. Download data (see [Challenge Website](http://medicalood.dkfz.de/web/)) to `./data/original`.
2. Save 2D slices along all axes
    ```bash
    python mood/utils/preprocessing/save_2D.py -i ./data/original/brain_train/ -o ./data/preprocessed/brain_train/2d_axis_0 -a 0
    python mood/utils/preprocessing/save_2D.py -i ./data/original/brain_train/ -o ./data/preprocessed/brain_train/2d_axis_1 -a 1
    python mood/utils/preprocessing/save_2D.py -i ./data/original/brain_train/ -o ./data/preprocessed/brain_train/2d_axis_2 -a 2
   ...
   ```
3. Optionally, create folds for cross-validation or **use ours folds** (`folds` dir)
    ```bash
    python mood/utils/preprocessing/create_folds.py -i ./data/original/brain_train/ -o ./folds/brain/train_folds_10.csv -n 10
    python mood/utils/preprocessing/create_folds.py -i ./data/original/abdom_train/ -o ./folds/abdom/train_folds_10.csv -n 10
   ```
4. Optionally: create a synthetic dataset for validation
    ```bash
    python mood/utils/data/create_val_dataset_with_synthetic_anomalies.py \
            -i ./data/original/brain_train/ \
            -o ./data/preprocessed/brain_train/3d_test \
            -m ./data/preprocessed/brain_train/3d_test_masks/ \
            --folds_path ./folds/brain/train_folds_10.csv
            --fold 0
   
   python mood/utils/data/create_val_dataset_with_synthetic_anomalies.py \
            -i ./data/original/abdom_train/ \
            -o ./data/preprocessed/abdom_train/3d_test \
            -m ./data/preprocessed/abdom_train/3d_test_masks/ \
            --folds_path ./folds/abdom/train_folds_10.csv
            --fold 0
   
   ```

## Training

### Pre-training of VGG19 features
    
Since no other data and data sources were allowed to use in the challenge,
we used an unsupervised learning framework [SimCLR](https://arxiv.org/abs/2002.05709)
to pre-train VGG19 features (used in the perceptual loss)

See [our fork](https://github.com/ninatu/SimCLR) of implementation 
of SimCLR adapted for the VGG19 training on provided data.

Save the pre-trained weights in `./output/vgg_weights/simclr_exp_1.tar`.
    

### Training Deep Perceptual Autoencoder

#### Example
See examples of configs for training and inference in `configs` dir.

To train Deep Perceptual Autoencoder (DPA), run:
```bash
python mood/main.py train ./configs/train_example.yaml
```

To inference and evaluate your model on synthetic dataset, run
```bash
python mood/main.py inference_evaluate_3d ./configs/inference_3d_example.yaml
```

#### Final Model

To train models as in our final submission, use the configs in `configs/brain/pixel`, `configs/brain/sample`,
`configs/abdom/pixel`, `configs/abdom/sample`

```bash
python mood/main.py train configs/brain/pixel/axis_0/train_config.yaml
python mood/main.py train configs/brain/pixel/axis_1/train_config.yaml
python mood/main.py train configs/brain/pixel/axis_2/train_config.yaml

python mood/main.py train configs/brain/sample/axis_0/train_config.yaml
python mood/main.py train configs/brain/sample/axis_1/train_config.yaml
python mood/main.py train configs/brain/sample/axis_2/train_config.yaml

...
```



## Building Docker With Final Model

Our final prediction was the average of predictions over three models 
in brain MRI task (applied along different axes), and over two models in abdominal CT task
(applied along 0'th and 2'th axes).

In order to build a docker with the final model:
1. Put your trained model into folder `submission_data`
2. Run 
```bash
docker build . -t mood:latest
```

Inference using the docker:
```bash
docker run --gpus all -v {input_dir}:/mnt/data -v {output_dir}:/mnt/pred mood:latest sh /workspace/run_{sample/pixel}_{TASK}.sh /mnt/data /mnt/pred
```

## Cite
If you use this code in your research, please cite:

```bibtex
@article{zimmerer2022mood,
  title={MOOD 2020: A public Benchmark for Out-of-Distribution Detection and Localization on medical Images},
  author={Zimmerer, David and Full, Peter M and Isensee, Fabian and J{\"a}ger, Paul and Adler, Tim and Petersen, Jens and K{\"o}hler, Gregor and Ross, Tobias and Reinke, Annika and Kascenas, Antanas and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE},
  volume={41},
  number={10},
  pages={2728-2738},
  doi={10.1109/TMI.2022.3170077}
}
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://github.com/ninatu/mood_challenge/blob/master/LICENSE

