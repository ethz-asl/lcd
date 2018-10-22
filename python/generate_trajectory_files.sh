#!/bin/bash
source ~/catkin_ws/devel/setup.bash
source ~/catkin_extended_ws/devel/setup.bash
SCENENETROSTOOLS_PATH=~/catkin_ws/src/scenenet_ros_tools/
PYSCENENET_PATH=/media/francesco/101f61e3-7b0d-4657-b369-3b79420829b8/francesco/ETH/Semester_3/Semester_Project/pySceneNetRGBD/
PYTHONSCRIPTS_PATH=~/catkin_extended_ws/src/line_tools/python/
LOCALLINESTXT_PATH=~/catkin_extended_ws/src/line_tools/data/train_lines/
TRAJ_NUM=${1:-1};
SHARE_PATH=${2:-/media/line_tools/fmilano/data/};
echo $TRAJ_NUM;
echo $SCENENETROSTOOLS_PATH;
cd $SCENENETROSTOOLS_PATH;
# Generate bag
echo -e "\n****  Generating bag for trajectory ${TRAJ_NUM} ****\n";
source ~/.virtualenvs/line_tools/bin/activate;
if [ -e ${PYSCENENET_PATH}/scenenet_traj_train_$TRAJ_NUM.bag ]
then
    echo 'Bag file already existent. Using bag found';
else
    rosrun scenenet_ros_tools scenenet_to_rosbag.py -scenenet_path $PYSCENENET_PATH -trajectory $TRAJ_NUM -output_bag ${PYSCENENET_PATH}/scenenet_traj_train_$TRAJ_NUM.bag;
fi
# Generate render paths file
cd $PYTHONSCRIPTS_PATH;
echo -e "\n**** Generating render paths ****\n";
python get_render_paths.py -scenenet_path $PYSCENENET_PATH;
# Create folders to store the data
echo -e "\n**** Creating folders to store the data for trajectory ${TRAJ_NUM} ****\n";
bash create_data_dir.sh $TRAJ_NUM;
rm ${LOCALLINESTXT_PATH}/traj_${TRAJ_NUM}/*
# Create mock file to handle first frame, that currently does not get detected
touch ${LOCALLINESTXT_PATH}/traj_${TRAJ_NUM}/lines_2D_0.txt;
touch ${LOCALLINESTXT_PATH}/traj_${TRAJ_NUM}/lines_2D_kept_0.txt;
touch ${LOCALLINESTXT_PATH}/traj_${TRAJ_NUM}/lines_with_labels_0.txt;
# Play bag and record data
roscd line_ros_utility;
echo -e "\n**** Playing bag and recording data for trajectory ${TRAJ_NUM} ****\n";
roslaunch line_ros_utility detect_cluster_show.launch trajectory:=${TRAJ_NUM} &
LAUNCH_PID=$!;
rosbag play ${PYSCENENET_PATH}/scenenet_traj_train_${TRAJ_NUM}.bag;
sudo kill ${LAUNCH_PID};
# Generate virtual camera images
cd $PYTHONSCRIPTS_PATH;
echo -e "\n**** Generating virtual camera images for trajectory ${TRAJ_NUM} ****\n";
source ~/.virtualenvs/line_tools/bin/activate;
python get_virtual_camera_images.py -trajectory ${TRAJ_NUM};
# Move everything to share folders
cd ${LOCALLINESTXT_PATH}
cd ..
echo -e "\n**** Moving files for trajectory ${TRAJ_NUM} to share folder ****\n";
mv train/traj_${TRAJ_NUM} ${SHARE_PATH}/train &
mv train/traj_${TRAJ_NUM}_inpaint ${SHARE_PATH}/train &
mv train_lines/traj_${TRAJ_NUM} ${SHARE_PATH}/train_lines &
wait;
# Delete bag
rm ${PYSCENENET_PATH}/scenenet_traj_train_$TRAJ_NUM.bag;
