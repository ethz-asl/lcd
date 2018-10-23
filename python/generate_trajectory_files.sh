#!/bin/bash
source ~/catkin_ws/devel/setup.bash
source ~/catkin_extended_ws/devel/setup.bash
SCENENETROSTOOLS_PATH=~/catkin_ws/src/scenenet_ros_tools/
PYSCENENET_PATH=/media/francesco/101f61e3-7b0d-4657-b369-3b79420829b8/francesco/ETH/Semester_3/Semester_Project/pySceneNetRGBD/
PYTHONSCRIPTS_PATH=~/catkin_extended_ws/src/line_tools/python/
LOCALLINESTXT_PATH=~/catkin_extended_ws/src/line_tools/data/train_lines/
TRAJ_NUM=${1:-1};
SHARE_PATH=${2:-/media/line_tools/fmilano/data/};
EULER_ADDRESS=fmilano@euler.ethz.ch

cd $SCENENETROSTOOLS_PATH;

# Stop the entire rather than a single command when Ctrl+C
trap "exit" INT

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
# Only create lines files if not there already (and valid)
if [ -e ${PYTHONSCRIPTS_PATH}/VALID_LINES_FILES_${TRAJ_NUM} ]
then
   echo 'Found valid lines files. Using them'
else
   rm ${LOCALLINESTXT_PATH}/traj_${TRAJ_NUM}/*

   # Play bag and record data
   roscd line_ros_utility;
   echo -e "\n**** Playing bag and recording data for trajectory ${TRAJ_NUM} ****\n";
   roslaunch line_ros_utility detect_cluster_show.launch trajectory:=${TRAJ_NUM} &
   LAUNCH_PID=$!;
   rosbag play -d 3.5 ${PYSCENENET_PATH}/scenenet_traj_train_${TRAJ_NUM}.bag;
   sudo kill ${LAUNCH_PID};
   cd $PYTHONSCRIPTS_PATH;
   # Creates a file to say that lines files have been generated, to avoid creating them
   # again if reexecuting the script before moving files
   touch ${PYTHONSCRIPTS_PATH}/VALID_LINES_FILES_${TRAJ_NUM};
fi

# Only generate virtual camera images files if not there already (and valid)
if [ -e ${PYTHONSCRIPTS_PATH}/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM} ]
then
   echo 'Found valid virtual camera images. Using them'
else
   # Generate virtual camera images
   cd $PYTHONSCRIPTS_PATH;
   echo -e "\n**** Generating virtual camera images for trajectory ${TRAJ_NUM} ****\n";
   source ~/.virtualenvs/line_tools/bin/activate;
   python get_virtual_camera_images.py -trajectory ${TRAJ_NUM};
   # Creates a file to say that virtual camera images have been generated, to avoid creating
   # them again if reexecuting the script before moving files
   touch VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM};
fi

# Move everything to share folders
cd ${LOCALLINESTXT_PATH}
cd ..
echo -e "\n**** Zipping files for trajectory ${TRAJ_NUM} ****\n";
tar -czf traj_${TRAJ_NUM}.tar.gz train/traj_${TRAJ_NUM} train_lines/traj_${TRAJ_NUM}
echo -e "\n**** Moving archive for trajectory ${TRAJ_NUM} ****\n";
#mv traj_${TRAJ_NUM}.tar.gz ${SHARE_PATH};
scp traj_${TRAJ_NUM}.tar.gz ${EULER_ADDRESS}:;
rm traj_${TRAJ_NUM}.tar.gz;
echo -e "\n*** Extracting archive for trajectory ${TRAJ_NUM} ****\n";
#tar -xzf ${SHARE_PATH}/traj_${TRAJ_NUM}.tar.gz -C ${SHARE_PATH};
#rm ${SHARE_PATH}/traj_${TRAJ_NUM}.tar.gz;
ssh ${EULER_ADDRESS} "tar -xvzf traj_${TRAJ_NUM}.tar.gz && rm traj_${TRAJ_NUM}.tar.gz"
# Delete lines files and virtual camera images
echo -e "\n**** Delete lines files for trajectory ${TRAJ_NUM} ****\n";
rm ${PYTHONSCRIPTS_PATH}/VALID_LINES_FILES_${TRAJ_NUM};
rm -r ${LOCALLINESTXT_PATH}/../train_lines/traj_${TRAJ_NUM};
echo -e "\n**** Delete virtual camera images for trajectory ${TRAJ_NUM} ****\n";
rm ${PYTHONSCRIPTS_PATH}/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM};
rm -r ${LOCALLINESTXT_PATH}/../train/traj_${TRAJ_NUM};

# Delete bag
echo -e "\n*** Deleting bag for trajectory ${TRAJ_NUM} ****\n";
rm ${PYSCENENET_PATH}/scenenet_traj_train_$TRAJ_NUM.bag;
