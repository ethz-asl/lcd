import numpy as np
import os
import sys

from get_protobuf_paths import get_protobuf_path


class SceneNetCameraToWorldMatrixRetriever:
    """ Retrieves a camera-to-world matrix from a frame in a trajectory in
        SceneNetRGBD.

    Args:
        trajectory (int): Number of the trajectory in the dataset dataset_name
            from SceneNetRGBD.
        dataset_name (string): Either 'val' or 'train_NUM', where 'NUM' is a
            number between 0 and 16.
        scenenetscripts_path (string): Path to the scripts from pySceneNetRGBD
            (e.g. '../pySceneNetRGBD/').

    Attributes:
        views: Set of views of the camera for each frame in the trajectory.
    """

    def __init__(self, trajectory, dataset_name, scenenetscripts_path):
        sys.path.append(scenenetscripts_path)
        import scenenet_pb2 as sn

        trajectories = sn.Trajectories()

        # Find protobuf file associated to dataset_name
        protobuf_path = get_protobuf_path(dataset_name)
        if protobuf_path is None:
            sys.exit('camera_utils.py: Error in retrieving protobuf_path.')

        try:
            with open(protobuf_path, 'rb') as f:
                trajectories.ParseFromString(f.read())
        except IOError:
            sys.exit('camera_utils.py: Scenenet protobuf data not found at '
                     'location: {}.\n'.format(protobuf_path) + 'Please ensure '
                     'you have copied the pb file to the data directory.')

        self.views = trajectories.trajectories[trajectory].views

    def get_camera_to_world_matrix(self, frame_id):
        """ Given a frame ID of a frame in the trajectory, returns the
            camera-to-world matrix.

        Args:
            frame_id (int): Frame ID of the frame in the input trajectory.

        Returns:
            camera_to_world_matrix (numpy array): Camera-to-world matrix.
        """
        from camera_pose_and_intrinsics_example import \
            camera_to_world_with_pose, interpolate_poses

        view = self.views[frame_id]
        ground_truth_pose = interpolate_poses(view.shutter_open,
                                              view.shutter_close, 0.5)
        camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)

        return camera_to_world_matrix


class SceneNNCameraToWorldMatrixRetriever:
    """ Retrieves a camera-to-world matrix from a frame in a trajectory in
        SceneNN.

    Args:
        trajectory (int): Number of the trajectory (scene) in SceneNN.
        dataset_path (string): Path containing the different image files from
            the dataset. In particular, this path should contain a subfolder XYZ
            for each scene (where XYZ is a three-digit ID associated to the
            scene, e.g. 005) and a subfolder 'intrinsic'. It is required here
            because the file with the trajectory poses is contained in there.

    Attributes:
        matrices: Set of camera-to-world matrices.
    """

    def __init__(self, trajectory, dataset_path):
        # Read file 'trajectory.log'.
        trajectory_file_path = os.path.join(
            dataset_path, '{:03d}'.format(trajectory), 'trajectory.log')
        # Initialize camera-to-world matrices.
        self.matrices = {}
        with open(trajectory_file_path, 'rb') as f:
            for i, line in enumerate(f):
                # NOTE: we discard frame 1, because ROS bags from SceneNN start
                # from frame 2.
                if i >= 5:
                    if i % 5 == 0:
                        curr_idx = i / 5 + 1
                        print("curr_idx is {}".format(curr_idx))
                        self.matrices[curr_idx] = np.empty([4, 4])
                    else:
                        print(line.split())

                        self.matrices[curr_idx][i % 5 - 1, :] = line.split()

    def get_camera_to_world_matrix(self, frame_id):
        """ Given a frame ID of a frame in the trajectory, returns the
            camera-to-world matrix.

        Args:
            frame_id (int): Frame ID of the frame in the input trajectory.

        Returns:
            camera_to_world_matrix (numpy array): Camera-to-world matrix.
        """
        try:
            camera_to_world_matrix = self.matrices[frame_id]
        except KeyError:
            raise KeyError("Please only use frame indices greater than 2 (cf."
                           "scenenn_ros_tools/scenenn_to_rosbag.py).")

        return camera_to_world_matrix
