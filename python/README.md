This package contains Python scripts that serve the following purposes:
1. Processing the lines data obtained from the package `line_ros_utility`, to obtain the virtual-camera images associated to the lines and storing all the data (in directories of .txt files) to later feed them to a neural network;
2. Training the neural network that will be used to cluster the lines;
3. Providing a script for the visualization of line clusterings;
4. Training the neural network for the description of line clusters;
5. Evaluating the full place recognition pipeline and visualizing the matched clusters.

## Requirements
Python 2.7 is used for the generation of data (due to ROS packages being used) and python 3.7 is used for the training, evaluation and visualization. You can create virtual environments and install the packages:
```bash
virtualenv --python=python2.7 ~/.virtualenvs/line_tools_2
source ~/.virtualenvs/line_tools/bin/activate
pip install -r requirements.txt
```
or
```bash
virtualenv --python=python2.7 ~/.virtualenvs/line_tools_3
source ~/.virtualenvs/line_tools/bin/activate
pip install -r requirements.txt
```
To use `jupyter notebook`, you need to type the following commands when you are inside the `line_tools` virtual environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name line_tools
```

Keras and tensorflow-gpu is used here. It is strongly recommended to use a GPU with sufficient memory for training. 

### Main scripts
- `generate_raw_data.py`: Runs all python scripts required for the generation and storage of the geometric information and virtual camera images. This script will automatically be run by the `generate_data.sh` script. The scripts include:
  - `interiornet_to_rosbag`: Publishes the InteriorNet dataset to the ROS node
  - `get_virtual_camera_images.py`: Obtains the virtual camera images of each line.
  - `split_dataset_framewise.py`: Currently, the InteriorNet dataset contains an error that some instances of the same semantic label are assigned the same instance label. This script fixes it (in a makeshift way), by splitting the instances with a mean shift algorithm. The radius can be set in the corresponding file. 

  _Arguments_:
  
  - `dataset_path`: The path where the dataset is stored.
  - `geometry_path: The path where the ROS node output is stored.
  - `vci_path`: The path where the virtual camera images are stored.
  - `output_path`: The path where the files used for training are stored.
  - `light_type`: Either 'random' or 'original', specifying the type of lighting used for the InteriorNet scenes.

The resulting scene folders can be copied and pasted into different folders to create a training and validation subset. Note that these subsets should not contain the same scenes, as this will distort the results. The scripts for training and evaluation of the neural networks are located in the `clustering_and_description` directory:


- `train_clustering.py`: Trains the neural-network for the clustering of lines. Please note that there are several training parameters including the path to the train and validation set, that can be changed by editing directly the associated variables in the script and that are documented within the script itself. A test directory needs to be specified with scenes where the output of the neural network can be visualized with `visualize_clusters.py`.

  _Arguments_:
  
  - `-pretrain`: If training the image encoding network. If not specified, the entire network will be trained. However, pretrained image encoding weights have to be provided;
  - `-model_checkpoint_dir`: If specified, the model checkpoint from past training is loaded. Epoch needs to be specified as well;
  - `-epoch`: The number of the epoch from past training;

- `visualize_clusters.py`: A script to visualize the outputs of the clustering network interactively. `train_clustering.py` generates clusterings of the test dataset that can be visualized with this script.

  _Arguments_:
  
  - `--path_to_results`: The path where the test results of the clustering network are to be stored.
 
 
- `train_descriptor.py`: Trains the neural-network for the description of clusters. Please note that there are several training parameters including the path to the train and validation set, that can be changed by editing the associated variables in the script directly and that are documented within the script itself.

  _Arguments_:
  
  - `-model_checkpoint_dir`: If specified, the model checkpoint from past training is loaded. Epoch needs to be specified as well;
  - `-epoch`: The number of the epoch from past training;


- `evaluate_pipeline.py`: The script to evaluate the place recognition performance of the full pipeline, the full pipeline with ground truth clustering and the SIFT bag-of-words approach on the validation set. Also, the clustering performance (NMI) of the clustering network and agglomerative clustering can be evaluated. The paths and variables need to be set within the script.


## Usage
**Data generation**
Generate the data by using the data generation script `generate_data.sh` and specifying the corresponding paths. If it is needed to change settings of the line detector ROS node, it can be run separately:
```bash
roslaunch line_ros_utility detect_and_save_lines.launch
```
and then run `generate_raw_data.py` with arguments explained above.

The data used for training is stored in the directory specified by `train_path`. It consists of scene folders containing all frames of a scene. After generating all data, create a directory for the train, validation and test set each. Split the data by moving a fraction of the folder to the train and validation set. Careful not to have duplicates in the train and validation directories. Scenes where visualization is desired can be copied into the test directory. 


**Training and testing**

Once the data folders have been created, the network can be trained.  
The first step is to pretrain the image encoding network. Set the paths to the data subsets in the python script `train_clustering.py` and run it with the `-pretrain` argument. After finishing training, the best epoch can be used by setting the path to the image encoding weights in the `train_clustering.py` script. Run it again without the `-pretrain` argument. The weights and training progress of each epoch are saved to the log directory. The progress can be viewed with tensorboard. 
The results of the clustering of the test set can be viewed with `visualize_clusters.py`.

The next step is to train the descriptor network by running `train_descriptor.py`. The weights of each epoch are again saved to the log directory. 

Finally, the full pipeline can be evaluated with `evaluate_pipeline.py`. There, the path to the best weights of the neural networks, and the path to the validation subset need to be specified. Also, some parameters can be tuned for the nearest neighbor matching approach and the SIFT bag-of-words method.
