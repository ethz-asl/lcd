"""
Get render path for each trajectory from protobuf file and saves it to file.
This is useful because when publishing the ROS bag associated to a trajectory
the number of the trajectory does not automatically correspond to the number of
the trajectory, so for instance trajectory with index 1 could map to the folder
0/784 in your data/train folder.
"""
import os
import sys
import argparse

def write_render_paths():
    trajectories = sn.Trajectories()
    # Read all trajectories from the protobuf file.
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(
            data_root_path))
        print(
            'Please ensure you have copied the pb file to the data directory')
    with open(output_path, 'w') as fileout:
        traj_count = 0;
        for traj in trajectories.trajectories:
            fileout.write('{0} {1}\n'.format(traj_count, traj.render_path))
            traj_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write SceneNet render paths to file.')
    parser.add_argument(
        "-scenenet_path",
        default="../pySceneNetRGBD",
        help="Path to the pySceneNetRGBD folder.")
    parser.add_argument(
        "-output_path",
        default="render_paths.txt",
        help="Path to the output text file with the render paths.")

    args = parser.parse_args()
    if args.scenenet_path:
        scenenet_path = args.scenenet_path
    if args.output_path:
        output_path = args.output_path

    # Include the pySceneNetRGBD folder to the path and import its modules.
    sys.path.append(scenenet_path)
    import scenenet_pb2 as sn

    data_root_path = os.path.join(scenenet_path, 'data/train')
    print('data_root_path is ' + data_root_path)
    protobuf_path = os.path.join(scenenet_path,
        'data/train_protobufs/scenenet_rgbd_train_0.pb')
    write_render_paths()
