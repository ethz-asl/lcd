#!/bin/bash
source ~/catkin_ws/devel/setup.bash
source ~/catkin_extended_ws/devel/setup.bash
source ~/.virtualenvs/line_tools/bin/activate
CURRENT_DIR=`dirname $0`
source $CURRENT_DIR/config_paths_and_variables.sh

# Check dataset name
if [ -z $DATASET_NAME ]
then
    echo "Please provide the name of dataset to use. Possible options are 'train' and 'val'. Exiting."
    exit 1
else
    # Check that name is valid
    case "$DATASET_NAME" in
        train)
            echo "Using training set from pySceneNetRGBD."
            ;;
        val)
            echo "Using validation set from pySceneNetRGBD."
            ;;
        *)
            echo "Invalid argument $DATASET_NAME. Valid options are 'train' and 'val'. Exiting."
            exit 1
            ;;
    esac
fi

# Retrieve correct protobuf path from dictionary
PROTOBUF_PATH=$(python "$PYTHONSCRIPTS_PATH"/tools/get_protobuf_paths.py -dataset_name $DATASET_NAME)
if [ -z "$PROTOBUF_PATH" ]
then
  echo "No entry for $DATASET_NAME was found in the protobuf dictionary (python/config_protobuf_paths). Exiting."
  exit 1
else
  echo "PROTOBUF_PATH is '$PROTOBUF_PATH"
fi

# Check protobuf path
if [ ! -f "$PROTOBUF_PATH" ]
then
    echo "Could not find protobuf file at $PROTOBUF_PATH, exiting."
    exit 1
fi

# Create output trajectories (if nonexistent)
mkdir -p "$BAGFOLDER_PATH"/${DATASET_NAME};
mkdir -p "$LINESANDIMAGESFOLDER_PATH";
mkdir -p "$TARFILES_PATH"/${DATASET_NAME};
mkdir -p "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/;

# Stop the entire rather than a single command when Ctrl+C
trap "exit" INT

# Generate bag
echo -e "\n****  Generating bag for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
if [ -e "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag ]
then
    # PLEASE NOTE! If an error occurs during the formation of the bag (e.g.
    # early termination by interrupt) the script will not check if the bag is
    # valid. Therefore invalid bags should be manually removed.
    echo 'Bag file already existent. Using bag found.';
else
    rosrun scenenet_ros_tools scenenet_to_rosbag.py -scenenet_path "$SCENENET_DATASET_PATH" -trajectory $TRAJ_NUM -output_bag "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag -protobuf_path "$PROTOBUF_PATH" -dataset_type ${DATASET_NAME};
fi

# Create folders to store the data
echo -e "\n**** Creating folders to store the data for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
bash "$CURRENT_DIR"/create_data_dir.sh $TRAJ_NUM "$LINESANDIMAGESFOLDER_PATH" $DATASET_NAME;
# Only create lines files if not there already (and valid)
if [ -e "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME} ]
then
   echo 'Found valid lines files. Using them.'
else
   # Delete any previous line textfile associated to that trajectory
   rm "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}/*
   # Play bag and record data

   echo -e "\n**** Playing bag and recording data for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
   roslaunch line_ros_utility detect_cluster_show.launch trajectory:=${TRAJ_NUM} write_path:="$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/ &
   LAUNCH_PID=$!;
   rosbag play -d 3.5 "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag;
   sudo kill ${LAUNCH_PID};

   # Creates a file to say that lines files have been generated, to avoid
   # creating them again if reexecuting the script before moving files
   touch "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME};
fi

# Only generate virtual camera images files if not there already (and valid)
if [ -e "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME} ]
then
   echo 'Found valid virtual camera images. Using them.'
else
   # Generate virtual camera images
   echo -e "\n**** Generating virtual camera images for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
   python "$PYTHONSCRIPTS_PATH"/get_virtual_camera_images.py -trajectory ${TRAJ_NUM} -scenenetscripts_path "$SCENENET_SCRIPTS_PATH" -dataset_name ${DATASET_NAME} -dataset_path "$SCENENET_DATASET_PATH"/data/${DATASET_NAME}/ -linesandimagesfolder_path "$LINESANDIMAGESFOLDER_PATH"/;
   # Creates a file to say that virtual camera images have been generated, to
   # avoid creating them again if reexecuting the script before moving files
   touch "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME};
fi

# Create archive
echo -e "\n**** Zipping files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
cd "$LINESANDIMAGESFOLDER_PATH";
# (the --no-name option is to prevent gzip to insert time info in the header, so that two archives containing the exact same files can in fact have same md5 hash)
tar -cf - ${DATASET_NAME}/traj_${TRAJ_NUM} ${DATASET_NAME}_lines/traj_${TRAJ_NUM} | gzip --no-name > "$TARFILES_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}.tar.gz

# Split dataset
echo -e "\n**** Splitting dataset for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
python "$PYTHONSCRIPTS_PATH"/split_dataset_with_labels_world.py -trajectory ${TRAJ_NUM} -protobuf_path "$PROTOBUF_PATH" -path_to_linesfiles "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/ -path_to_virtualcameraimages "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}/ -output_path "$LINESANDIMAGESFOLDER_PATH" -scenenetscripts_path "$SCENENET_SCRIPTS_PATH"
echo -e "\n**** Pickling files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
python "$PYTHONSCRIPTS_PATH"/pickle_files.py -splittingfiles_path "$LINESANDIMAGESFOLDER_PATH" -output_path "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/ -dataset_name ${DATASET_NAME}
for word in test train val all_lines; do
  # NOTE: These files include absolute paths w.r.t to the machine where the
  # data was generated. Still, they are included as a reference to how pickled
  # files were formed (i.e., how data was divided before pickling the data).
  # Further conversion is needed when using these files to train the NN, to
  # replace the absolute path.
  mv "$LINESANDIMAGESFOLDER_PATH"/${word}.txt "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/;
done

# Delete lines files and virtual camera images
echo -e "\n**** Delete lines files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
rm "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME};
rm -r "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM};
echo -e "\n**** Delete virtual camera images for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
rm "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME};
rm -r "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM};
