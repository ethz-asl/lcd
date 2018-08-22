# Preprocess and train data in tensorflow

This repository is for
1. preprocessing the lines data got from ros package `line_ros_utility`
2. training with triplet loss to get the embeddings for lines
3. visualization

## Requirements
Python 2.7 is used here. You can create a virtual environment and install the packages:
```bash
virtualenv --python=python2.7 ~/.virtualenvs/line_tools
source ~/.virtualenvs/line_tools/bin/activate
pip install -r requirements.txt
```
To use jupyter notebook, you need to type the following commands when you are inside `line_tools` virtual environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name line_tools
```

Tensorflow cpu version is used here. If you want to use gpu, change `requirements.txt` accordingly.

## Usage
Before reaching to this repository, make sure that you have followed the instructions for the ros packages and get the lines data in `../data/train_lines`.

Here we use the trajectory 1 in the [train_0](https://robotvault.bitbucket.io/scenenet-rgbd.html) dataset of `SceneNetRGBD` as an example. You first need to clone the repository [pySceneNetRGBD](https://github.com/jmccormac/pySceneNetRGBD), download the dataset as well as the protobuf to `pySceneNetRGBD/data`. Set `pySceneNetRGBD_root` in `tools/pathconfig.py` properly.

If you want to try other trajectories, change the variables `path_to_photos` and `path_to_lines` accordingly.

You can create subdirectories in `../data/train` to store the images data we will use for training.
```bash
./create_data_dir.sh
```

Run `get_virtual_camera_images.py` to get the virtual camera image associated to each line for every frame in trajectory 1.
```bash
python get_virtual_camera_images.py
```

Notice that these images data are reprojection of the pointcloud from a different viewpoint and there are black parts in the images that have no data. We can further more inpaint these images to fill in the "small" black parts. To inpaint, check the notebook `examples/00-Go_through_our_data.ipynb`.

Take trajectory 1 as an example, we split its 300 frames to `train`, `val` and `test` (ratio = 3:1:1). (Normally one can follow the split of dataset provided by `SceneNetRGBD`)
```bash
python split_dataset_with_labels_world.py
```

The path to image data and label(4d array) of each image are then be written into the file `train.txt`, `val.txt` and `test.txt`.

The label is a 4d array `[center of the line, instance label of the line]`

The last thing to do before training is to download the pretrained weights [bvlc_alexnet.npy](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) for **AlexNet** and place it in the folder `model`:

Until now we should have all the data needed, to
train with **AlexNet** and **triplet loss**:
```bash
python train.py
```
**Note**: If you use a different train set, you need to calculate the train set mean with `tools/train_set_mean.py` and change `mean = np.array([22.47429166, 20.13914579, 5.62511388])` to your new mean in `model/datagenerator.py`

The checkpoint for 30 epoches is stored in `logs`.

## Notebooks
Notebooks provide better comprehension of the codes as well as some visualizations. One can check them in `examples`.
