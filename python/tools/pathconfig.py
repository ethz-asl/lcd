import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))

pySceneNetRGBD_root = os.path.join(file_dir, '../../../pySceneNetRGBD/')
sys.path.insert(0, pySceneNetRGBD_root)

protobuf_path = os.path.join(
    pySceneNetRGBD_root, 'data/train_protobufs/scenenet_rgbd_train_0.pb')

# Path to lines data in the trajectory
path_to_lines_root = os.path.join(file_dir, '../../data/train_lines/traj_1')

# Path to the virtual camera images of lines
path_to_lines_image = os.path.join(file_dir, '../../data/train/traj_1/')
