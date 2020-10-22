FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install --upgrade pip && \
    pip install \
            numpy \
            setuptools \
            pyyaml \
            opencv-python \
            tensorflow-gpu \
            Pillow \
            h5py \
            keras \
            matplotlib \
            pandas \
            pydicom \
            scikit-image \
            scikit-learn \
            scipy \
            seaborn \
            tensorboard \
            tensorboardX \
            tensorflow \
            tensorflow-estimator \
            tqdm \
            nibabel

RUN pip install monai
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# copy files

RUN mkdir /workspace/mood && \
    mkdir /workspace/configs && \
    mkdir /workspace/data

COPY submission_data/data /workspace/data
COPY submission_data/configs /workspace/configs
ADD submission_data/scripts /workspace/

COPY mood /workspace/mood/mood
ADD setup.py /workspace/mood
RUN pip install /workspace/mood

RUN chmod +x /workspace/*.sh && \
    mkdir /mnt/data && \
    mkdir /mnt/pred
