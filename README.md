# Multi-modal learing using Tacotron

An implementation of modified version of the Tacotron speech synthesis model in TensorFlow to generate bird-chirps (audio) samples given an image of a bird.


## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download dataset.**

   Use this [link](https://link_todo.com) to download the dataset.
    * Unzip the downloaded file.
    * Setup the training data in the following structure:-
```
tacotron (project dir)
    |- training
         |- vgg19
         |    |- vgg19.npy
         |- bird-00001.npy
         |- bird-00002.npy
         |- ...
         |- chirp-mel-00001.npy
         |- chirp-mel-00002.npy
         |- ...
         |- chirp-spec-00001.npy
         |- chirp-spec-00002.npy
         |- ...
         |- train.txt
   ```

2. **Train model**
   ```
   python3 train.py
   ```

5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Test your model**
   ```
   python3 test.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000 --image_path ~/path/to/input/image
   ```


## Create your own Dataset
1. **Set up the followinng directory structure given raw images and audio**
```
    ├── raw_data
          ├── imgs              # Folder contains all the images of the birds from 6 different breeds
          │   ├── 0             # Duck
          │   ├── 1             # Hawk
          │   ├── 2             # Owl
          │   ├── 3             # Seagull
          │   └── 4             # Macaw
          │   └── 5             # Rooster
          └── wavs              # Folder contains all the sounds of the birds from 6 different breeds  
              ├── 0             # Duck
              ├── 1             # Hawk
              ├── 2             # Owl
              ├── 3             # Seagull
              └── 4             # Macaw
              └── 5             # Rooster
```

2. **Data generation**

    * Move raw data into specific directories and create metadata.

    ```
    python3 dataset_generator.py
    ```

    * Download pretrained [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) to your project root directory.

    * Preprocess data and generate dataset.

   ```
    python3 preprocess.py
    ```
