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
[Please note that the entire process of generating the bag files, obtaining the
lines data and performing the subsequent steps explained below can be
automatically executed by launching `roscore` and then running the script `generate_and_upload_to_share.sh` or separately the two scripts `generate_trajectory_files.sh` and `split_all_trajectories.sh`. Please look at
the content of the scripts for a more detailed explanation.]

Before reaching to this repository, make sure that you have followed the instructions for the ros packages and get the lines data in `../data/train_lines`.

Here we use the trajectory 1 in the [train_0](https://robotvault.bitbucket.io/scenenet-rgbd.html) dataset of `SceneNetRGBD` as an example. You first need to clone the repository [pySceneNetRGBD](https://github.com/jmccormac/pySceneNetRGBD), download the dataset as well as the protobuf to `pySceneNetRGBD/data`. Set `pySceneNetRGBD_root` in `tools/pathconfig.py` properly.
```bash
cd ../..
git clone https://github.com/jmccormac/pySceneNetRGBD.git
cd pySceneNetRGBD
mkdir data && cd data
wget http://www.doc.ic.ac.uk/~ahanda/train_split/train_0.tar.gz train_0.tar.gz
wget http://www.doc.ic.ac.uk/~ahanda/train_protobufs.tar.gz train_protobufs.tar.gz
tar -xvzf train_0.tar.gz train_protobufs.tar.gz
cd .. && make
```

If you want to try other trajectories, change the variables `path_to_photos` and `path_to_lines` accordingly.

You can create subdirectories in `../data/train` to store the images data we will use for training.
```bash
./create_data_dir.sh
```
Note that the above bash script also accepts the number of the trajectory as an
argument (1 by default).

Since the indices associated with the trajectories do not correspond to the
names of the folder in which the images are located in the dataset (e.g. trajectory with index 1 could map to the folder 0/784 in the data/train folder),
it is necessary to generate a text file containing the correspondences between trajectory numbers and the so called 'render path'. Do so by running `get_render_paths.py`.
```bash
python get_render_paths.py
```

Run `get_virtual_camera_images.py` to get the virtual camera image associated to each line for every frame in the input trajectory (1 by default).
```bash
python get_virtual_camera_images.py -trajectory 1
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
**Note**: If you use a different train set, you need to calculate the train set mean with `tools/train_set_mean.py` and pass the train set mean to the `ImageDataGenerator`.

## Notebooks
Notebooks provide better comprehension of the codes as well as some visualizations. One can check them in `examples`.

## Possible improvements
1. The virtual image for line is taken with the same camera as the SceneNetRGBD dataset's one. The camera's distance to the line is set to 3 meters, which introduce some occlusions for certain lines. As an alternative, one might consider using strange cameras, eg. omnidirectional camera. Thus we can put the "camera" near the line and project the pointcloud(or points near the line) to get virtual image for that line.

2. For an intersection line, there is just one instance label associated to it. It's probably better to associate it with two virtual images and two labels using the left&right sides planes of the line. This can be done in the `line_detection` library similarly to the function `LineDetector::assignColorToLines`.
