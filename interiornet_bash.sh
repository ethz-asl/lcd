source devel/setup.bash

Line detection:
roslaunch line_ros_utility detect_and_save_lines.launch write_path:=/home/felix/line_ws/data/line_tools/interior_lines/ scenenet_true_scenenn_false:=true interiornet:=true

InteriorNet to ROS:
rosrun interiornet_to_rosbag interiornet_to_rosbag.py --scene-path /home/felix/line_ws/data/interiorNet/data/HD7/3FO4IDEI1LAV_Dining_room --output-bag-path /home/felix/line_ws/data/bags/HD7/3FO4IDI9FO3C_Guest_room.bag --frame-step 1  --light-type original --traj 1 --publish

Generate virtual images:
python get_virtual_camera_images.py -linesandimagesfolder_path /home/felix/line_ws/data/line_tools/ -dataset_name interiornet -interiornet_scene_path /home/felix/line_ws/data/interiorNet/data/HD7/3FO4IDI9FO3C_Guest_room -interiornet_light_type original -interiornet_trajectory 1
