import sys
import rospy
import glob
import os

from std_msgs.msg import String

sys.path.insert(0, "/home/felix/line_ws/src/interiornet_to_rosbag/nodes")
import interiornet_to_rosbag
import get_virtual_camera_images
import split_dataset_framewise

if __name__ == '__main__':
    path_to_data = "/nvme/datasets/interiornet/*"
    line_path_master = "/nvme/line_ws/line_files"
    vci_path_master = "/nvme/line_ws/virtual_camera_images"
    train_path_master = "/nvme/line_ws/all_data"

    rospy.init_node('interiornet_node', anonymous=True)

    path_publisher = rospy.Publisher('/line_tools/output_path', String, queue_size=5)
    publishers = interiornet_to_rosbag.init_publishers()

    num_files = 0
    for dir_path in glob.iglob(path_to_data):
        if "Bathroom" in dir_path:
            continue

        scene_id = dir_path.split('/')[-1]

        print("Processing scene: {}".format(scene_id))

        line_path = os.path.join(line_path_master, scene_id)
        if not os.path.exists(line_path):
            os.mkdir(line_path)

        path_publisher.publish(line_path + "/")

        file_count = len([name for name in os.listdir(line_path)
                         if os.path.isfile(os.path.join(line_path, name))])
        if file_count < 60:
            interiornet_to_rosbag.convert(
                scene_path=dir_path,
                scene_type=7,
                light_type='original',
                traj=0,
                frame_step=1,
                to_frame=20000,
                output_bag='none',
                publishers=publishers,
                publish=True)

        remake_vcis = False
        vci_path = os.path.join(vci_path_master, scene_id)
        if not os.path.exists(vci_path) or not len(os.listdir(os.path.join(vci_path, "frame_19", "rgb"))) > 0:
            remake_vcis = True

            os.mkdir(vci_path)
            for i in range(20):
                os.mkdir(os.path.join(vci_path, 'frame_{}'.format(i)))
                os.mkdir(os.path.join(vci_path, 'frame_{}'.format(i), 'rgb'))
                os.mkdir(os.path.join(vci_path, 'frame_{}'.format(i), 'depth'))

            get_virtual_camera_images.get_virtual_camera_images_interiornet(
                scene_path=dir_path,
                scene_type=7,
                trajectory=0,
                light_type='original',
                linesfiles_path=line_path,
                output_path=vci_path,
            )

        scene_out_path = os.path.join(train_path_master, scene_id)
        if not os.path.exists(scene_out_path):
            os.mkdir(scene_out_path)

            train_path = os.path.join(scene_out_path, scene_id + "_frame_{}")
            # if remake_vcis or not os.path.exists(train_path.format(19)):
            split_dataset_framewise.split_scene(line_path, vci_path, train_path)



