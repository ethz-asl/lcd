This package contains Python scripts that serve the following purposes:
1. Processing the lines data obtained from the package `line_ros_utility`, to obtain the virtual-camera images associated to the lines and compacting storing all the data (in _pickle_ files) to later feed them to a neural network;
2. Training the neural network that will be used to retrieve the embeddings for the lines;
3. Providing utilities for visualization and statistics about the virtual-camera images and the lines.

## Requirements
Python 2.7 is used here. You can create a virtual environment and install the packages:
```bash
virtualenv --python=python2.7 ~/.virtualenvs/line_tools
source ~/.virtualenvs/line_tools/bin/activate
pip install -r requirements.txt
```
To use `jupyter notebook`, you need to type the following commands when you are inside `line_tools` virtual environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name line_tools
```

Tensorflow CPU version is used here. If you want to use GPU, change `requirements.txt` accordingly.

### Main scripts
*NOTE: the arguments of the scripts below (apart from those in `tools/` can be avoided by using the data-generation script `../generate_trajectory_files.sh`, after setting the variables in `../config_paths_and_variables.sh`*.
- `display_line_with_points_and_planes.py`: Displays a line in 3D, before and after readjustment, together with the points that are inliers to the planes fitted around it. The data for the line is read from `../line_with_points_and_planes.yaml`. The script is called by the `line_detection` package when the associated flag is set to `true`;


- `get_virtual_camera_images.py`: Obtain the virtual-camera images associated to lines extracted from a trajectory from either SceneNetRGBD or SceneNN.

  _Arguments_:
  - `-trajectory`: Number (index) of the trajectory in the original dataset;
  - `-frame_step`: Number of frames in one step of the ROS bag used to detect lines, i.e., (`frame_step` - 1) frames were originally skipped after each frame inserted in the ROS bag;
  - `-end_frame`: Index of the last frame in the trajectory;
  - `-scenenetscripts_path`: Path to folder containing the scripts from pySceneNetRGBD, in particular `scenenet_pb2.py` (e.g. `scenenetscripts_path`=`'pySceneNetRGBD/'`). Needed to extract the model of the virtual camera;
  - `-dataset_path`: Path to folder containing the different image files from the dataset. For SceneNetRGBD datasets, the path should be such that concatenating the render path to it gives a folder with folders `depth`, `instances` and `photo` inside (e.g. `dataset_path`=`'pySceneNetRGBD/data/train/'`). For SceneNN datasets, the path should contain a subfolder `XYZ/` for each scene (where `XYZ` is a three-digit ID associated to the scene, e.g. `005`) and a subfolder `intrinsic`.;
  - `-dataset_name`: If the data comes from the `val` or `train_NUM` dataset of SceneNetRGBD, either `'val'` or `'train_NUM'` (`NUM` is a number between 0 and 16). If the data comes from SceneNN, `'scenenn'`;
  - `-linesandimagesfolder_path`: Path to folder (e.g. `'data/'`) containing text lines files (e.g. under `'data/train_0_lines'`) and that should store the output virtual-camera images (e.g. under `'data/train_0'`);
  - `-output_path`: Data folder where to store the virtual-camera images (e.g. `'data/train_0'`).

  **NOTE:** the virtual-camera images obtained are reprojection of the point cloud from a different viewpoint and therefore always have 'black parts', i.e., regions of pixels for which no data are available. The following methods are currently available to make this problem milder:
  - Vary the distance of the virtual camera from the lines, by setting the variable `distance` (in meters). The closer the camera, the more empty pixels will be in the virtual-camera images, as the point cloud will appear more spread-out;
  - Only keep lines that have at least a certain fraction of nonempty pixels, by setting the variable `min_fraction_nonempty_pixels` (between 0 and 1);
  - _Impaint_ the empty pixels, i.e., reconstruct their intensity information by using the information from the surrounding nonempty pixels => Set the variable `impainting` to `True`. *Beware: this process is highly computationally expensive (up to 2 seconds per line on a regular laptop with i7 process @ 2.5 GHz).*


- `pickle_files.py`: Creates *pickle* files that compactly store the information and the virtual-camera image for each line in each frame, based on the splitting performed by `split_dataset_with_labels_world.py`.

  _Arguments_:
  - `-splittingfiles_path`: Path to the files indicating the splitting (i.e. {`train`, `test`, `val`}`.txt`);
  - `-output_path`: Path where to store the pickle files;
  - `-dataset_name`: If the data comes from the `val` or `train_NUM` dataset of SceneNetRGBD, either `'val'` or `'train_NUM'` (`NUM` is a number between 0 and 16). If the data comes from SceneNN, `'scenenn'`.


- `split_dataset_with_labels_world.py`: Converts the 3D lines detected from camera-frame coordinates to world-frame coordinates, by retrieving the camera-to-world matrix from the original dataset. It then splits the input data in a training, test and validation set.

  _Arguments_:
  - `-trajectory`: Number (index) of the trajectory in the original dataset;
  - `-frame_step`: Number of frames in one step of the ROS bag used to detect lines, i.e., (`frame_step` - 1) frames were originally skipped after each frame inserted in the ROS bag;
  - `-end_frame`: Index of the last frame in the trajectory;
  - `-scenenetscripts_path`: Path to folder containing the scripts from pySceneNetRGBD, in particular `scenenet_pb2.py` (e.g. `scenenetscripts_path`=`'pySceneNetRGBD/'`);
  - `-dataset_name`: If the data comes from the `val` or `train_NUM` dataset of SceneNetRGBD, either `'val'` or `'train_NUM'` (`NUM` is a number between 0 and 16). If the data comes from SceneNN, `'scenenn'`;
  - `-linesandimagesfolder_path`: Path to folder (e.g. `'data/'`) containing text lines files (e.g. under `'data/train_0_lines'`) as well as the virtual-camera images (e.g. under `'data/train_0'`);
  - `-output_path`: Path where to write the `.txt` files with the splitting;
  - `-dataset_path`: Path to folder containing the different image files from the dataset. For SceneNetRGBD datasets, the path should be such that concatenating the render path to it gives a folder with folders `depth/`, `instances/` and `photo/` inside (e.g. `dataset_path`=`'pySceneNetRGBD/data/train/'`). For SceneNN datasets, the path should contain a subfolder `XYZ/` for each scene (where `XYZ` is a three-digit ID associated to the scene, e.g. `005`) and a subfolder `intrinsic`.


- `train.py`: Trains the neural-network that will later be used to retrieve the embeddings. Please note that there are several training parameters that can be changed by editing directly the associated variables in the script and that are documented within the script itself.

  _Arguments_:
  - `-job_name`: Name to assign to the training job (used to name the folder where the output will be stored).


- `test.py`: Retrieves the embeddings for a test set, by using a previously-trained model. It then clusters the embeddings based on the method chosen in the script itself and saves the embeddings to file. Please several parameters that can be changed by editing directly the associated variables in the script and that are documented within the script itself.


- `tools/histogram_empty_pixels.py`: Creates a histogram with the number of nonempty pixels in the virtual-camera images. The number of nonempty pixels is previously saved to disk by `get_virtual_camera_images.py`.

  _Arguments_:
  - `-textfile_path`: Path of the textfile containing the number of nonempty pixels for each virtual-camera images in the set considered.


- `tools/histogram_instances_in_pickle_file.py`: Creates a histogram of the distribution of the instance labels in a pickle file.

  _Arguments_:
  - `-picklefile_path`: Path of the pickle file;
  - `-save_data_path`: Path where to save the histogram statistics file.


- `tools/pickle_dataset.py`: Contains utils to pickle files and merge previously-creted pickle files, but also allows to _repickle_ files, by either keeping a restricted set of instances, or by substituting the virtual-camera images with empty images (for testing purposes).


- `tools/statistics_on_embeddings.py`: Displays statistics about the embeddings obtained for each instance in a certain dataset, i.e., it displays statistics about the potential clusters  that can be formed in the embedding space (e.g., mean and standard deviation  of the distances from the 'mean embedding', max and min 'intra-instance' distance, etc.).

  _Arguments_:
  - `-embeddings_file_path`: Path of the embeddings file;
  - `-instances_file_path`: Path of the ground-truth instance file;
  - `-output_file_path`: Path where to save the statistics.


## Usage
**Data generation**

Please note that the entire process of generating the bag files, obtaining the
lines data and virtual-camera images, splitting the dataset and generating pickle files can be
automatically executed by launching `roscore` and then running `generate_trajectory_files.sh`. Before generating the data, the variables in the configuration file `config_paths_and_variables.sh` should be set as indicated in the description in the script itself. *The latter is the suggested procedure.*

If you instead want to separately run each Python script in the pipeline without using the Bash script `generate_trajectory_files.sh`, please note that they all accept arguments that should be comply to a specific format. If some arguments are missing, their value will be replaced based on the paths and variables in `config_paths_and_variables.sh`.  
Once the lines data have been obtained (by using the procedure in `line_ros_utility`), the pipeline proceeds as follows:
- Generate the virtual-camera images => run the script `get_virtual_camera_images.py` with the arguments explained above;
- Convert the lines from camera-frame coordinates to world-frame coordinates and split the dataset in training, test and validation set => run the script `split_dataset_with_labels_world.py` with the arguments explained above;
- Compactly store all the data in _pickle_ files, to later feed them to the neural network => run the script `pickle_dataset.py` with the arguments explained above.

**Training and testing**

Once the pickle files have been generated, the network can be trained.  
The last thing to do before training is to download the pretrained weights [bvlc_alexnet.npy](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) for **AlexNet** and place it in the folder `model/`.

The training process can then be started by editing the script `train.py` to set the training parameters as well as the training set to use and then run it:
```python2
python train.py
```

Testing can be performed in the same way, by editing the script `test.py`, so as to select the previously-trained model and test set to use as well as the clustering strategy, and then running it:
```python2
python test.py
```


## Notebooks
Jupyter notebooks are available, to provide example execution of the code in separate steps, as well as some visualization. However, they are not meant to be comprehensive. You can check them in the `examples/` folder.
