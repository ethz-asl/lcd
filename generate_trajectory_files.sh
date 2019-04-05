#!/bin/bash
source ~/line_ws/devel/setup.bash
#source ~/.virtualenvs/line_tools/bin/activate
CURRENT_DIR=`dirname $0`
source $CURRENT_DIR/config_paths_and_variables.sh

# Check the trajectory number.
if [ -z $TRAJ_NUM ]
  then
    echo "Please provide the trajectory number (in config_paths_and_variables.sh).";
    exit 1
fi

# Check dataset name.
if [ -z $DATASET_NAME ]
then
    echo "Please provide the name of dataset to use (in config_paths_and_variables.sh). Possible options are and 'val' and 'train_NUM', where NUM is a number between 0 and 16, 'scenenn'. Exiting."
    exit 1
fi

# Create output directories (if nonexistent).
mkdir -p "$BAGFOLDER_PATH"/${DATASET_NAME};
mkdir -p "$LINESANDIMAGESFOLDER_PATH";
mkdir -p "$TARFILES_PATH"/${DATASET_NAME};
mkdir -p "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/;

# Stop the entire rather than a single command when Ctrl+C.
trap "exit" INT

# Set a flag depending on whether the dataset belongs to SceneNetRGBD or
# SceneNN.
# Check that name of the dataset is valid
case ${DATASET_NAME%_*} in
    train)
        if [ ${DATASET_NAME#*_} -ge 0 ] && [ ${DATASET_NAME#*_} -le 16 ]
        then
          SCENENET_TRUE_SCENENN_FALSE=true
        else
          echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16, and 'scenenn'."
          exit 1
        fi
        ;;
    val)
        SCENENET_TRUE_SCENENN_FALSE=true
        ;;
    scenenn)
        SCENENET_TRUE_SCENENN_FALSE=false
        ;;
    *)
        echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16, and 'scenenn'."
        exit 1
        ;;
esac

# Generate bag.
if [ "$SCENENET_TRUE_SCENENN_FALSE" = true ]
then
  # All frames are extracted from SceneNetRGBD trajectories.
  FRAME_STEP=1
  if [ -e "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag ]
  then
      echo "Bag file for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set already exists. Using bag found.";
  else
      echo -e "\n****  Generating bag for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
      echo -e "PLEASE NOTE! If an error occurs during the formation of the bag (e.g. "
      echo -e "early termination by interrupt) the script will not check if the bag "
      echo -e "is valid. Therefore invalid bags should be manually removed.\n"
      rosrun scenenet_ros_tools scenenet_to_rosbag.py --scenenet-path "$SCENENET_DATASET_PATH" --trajectory $TRAJ_NUM --dataset-type ${DATASET_TYPE} --train-set-split ${TRAIN_NUM} --output-bag "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag;
  fi
  # Get number of frames in the bag.
  NUM_FRAMES_IN_BAG=$(python "$PYTHONSCRIPTS_PATH"/tools/get_number_of_frames_in_bag.py -rosbag_path "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag);
  START_FRAME=0
else
  FRAME_STEP=25
  if [ -e "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenn_traj_${TRAJ_NUM}.bag ]
  # Due to the higher frame rate, only one every 25 frames is extracted from the
  # SceneNN trajectories.
  then
      echo "Bag file for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set already exists. Using bag found.";
  else
      echo -e "\n****  Generating bag for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
      echo -e "PLEASE NOTE! The process will take a long time. If an error occurs "
      echo -e "during the formation of the bag (e.g. early termination by interrupt) "
      echo -e "the script will not check if the bag is valid. Therefore invalid "
      echo -e "bags should be manually removed.\n"
      rosrun scenenn_ros_tools scenenn_to_rosbag.py -scenenn_data_folder "$SCENENN_DATASET_PATH" -frame_step ${FRAME_STEP} -scene_id $TRAJ_NUM -output_bag "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenn_traj_${TRAJ_NUM}.bag;
  fi
  # Get number of frames in the bag.
  NUM_FRAMES_IN_BAG=$(python "$PYTHONSCRIPTS_PATH"/tools/get_number_of_frames_in_bag.py -rosbag_path "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenn_traj_${TRAJ_NUM}.bag);
  # ROS bags from SceneNN start from frame 2 (cf. scenenn_ros_tools/scenenn_to_rosbag.py).
  START_FRAME=2
fi

echo -e "The bag contains ${NUM_FRAMES_IN_BAG} frames."
END_FRAME=$(($START_FRAME + ($NUM_FRAMES_IN_BAG - 1) * $FRAME_STEP))

# Create folders to store the data.
echo -e "\n**** Creating folders to store the data for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
bash "$CURRENT_DIR"/create_data_dir.sh $TRAJ_NUM "$LINESANDIMAGESFOLDER_PATH" $DATASET_NAME $START_FRAME $END_FRAME $FRAME_STEP;
# Only create lines files if not there already (and valid).
if [ -e "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME} ]
then
   echo 'Found valid lines files. Using them.'
else
  while
    # Delete any previous line textfile associated to that trajectory.
    if [ -d "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}/ ]
    then
      rm "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}/*
    fi
    # Play bag and record data.
    echo -e "\n**** Playing bag and recording data for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
    roslaunch line_ros_utility detect_and_save_lines.launch trajectory:=${TRAJ_NUM} write_path:="$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/ scenenet_true_scenenn_false:=$SCENENET_TRUE_SCENENN_FALSE start_frame:=${START_FRAME} frame_step:=${FRAME_STEP} &
    LAUNCH_PID=$!;
    if [ "$SCENENET_TRUE_SCENENN_FALSE" = true ]
    then
      rosbag play -d 3.5 --queue 300 -r 10 "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenet_traj_${TRAJ_NUM}.bag;
    else
      rosbag play -d 3.5 --queue 1000 -r 2 "$BAGFOLDER_PATH"/${DATASET_NAME}/scenenn_traj_${TRAJ_NUM}.bag;
    fi
    kill ${LAUNCH_PID};
    # Repeat process from scratch if not all lines have been generated.
    [ ! -e "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}/lines_2D_${END_FRAME}.txt ]
  do
    :
  done
  # Creates a file to say that (all) lines files have been generated, to avoid
  # creating them again if reexecuting the script before moving files.
  touch "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME};
fi

# Only generate virtual camera images files if not there already (and valid).
if [ -e "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME} ]
then
   echo 'Found valid virtual camera images. Using them.'
else
   # Generate virtual camera images.
   echo -e "\n**** Generating virtual camera images for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
   if [ "$SCENENET_TRUE_SCENENN_FALSE" = true ]
   then
     python "$PYTHONSCRIPTS_PATH"/get_virtual_camera_images.py -trajectory ${TRAJ_NUM} -scenenetscripts_path "$SCENENET_SCRIPTS_PATH" -dataset_name ${DATASET_NAME} -dataset_path "$SCENENET_DATASET_PATH"/data/${DATASET_NAME%_*}/ -linesandimagesfolder_path "$LINESANDIMAGESFOLDER_PATH"/;
   else
     python "$PYTHONSCRIPTS_PATH"/get_virtual_camera_images.py -trajectory ${TRAJ_NUM} -scenenetscripts_path "$SCENENET_SCRIPTS_PATH" -dataset_name ${DATASET_NAME} -dataset_path "$SCENENN_DATASET_PATH" -linesandimagesfolder_path "$LINESANDIMAGESFOLDER_PATH"/ -frame_step ${FRAME_STEP} -end_frame ${END_FRAME};
   fi

   # Creates a file to say that virtual camera images have been generated, to
   # avoid creating them again if reexecuting the script before moving files.
   touch "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME};
fi

# Create archive.
echo -e "\n**** Zipping files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
cd "$LINESANDIMAGESFOLDER_PATH";
# (the --no-name option is to prevent gzip from inserting time info in the
# header, so that two archives containing the exact same files can in fact have
# same md5 hash)
if [ -f "$TARFILES_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}.tar.gz ]
then
  echo -e "Archive file "$TARFILES_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}.tar.gz is already existent. Please delete it or rename, and relaunch this script to generate a new one."
  exit 1
else
  tar -cf - ${DATASET_NAME}/traj_${TRAJ_NUM} ${DATASET_NAME}_lines/traj_${TRAJ_NUM} | gzip --no-name > "$TARFILES_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}.tar.gz
fi

# Split dataset.
echo -e "\n**** Splitting dataset for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
if [ "$SCENENET_TRUE_SCENENN_FALSE" = true ]
then
  python "$PYTHONSCRIPTS_PATH"/split_dataset_with_labels_world.py -trajectory ${TRAJ_NUM} -linesandimagesfolder_path "$LINESANDIMAGESFOLDER_PATH" -output_path "$LINESANDIMAGESFOLDER_PATH" -scenenetscripts_path "$SCENENET_SCRIPTS_PATH" -dataset_name ${DATASET_NAME}
else
  python "$PYTHONSCRIPTS_PATH"/split_dataset_with_labels_world.py -trajectory ${TRAJ_NUM} -linesandimagesfolder_path "$LINESANDIMAGESFOLDER_PATH" -output_path "$LINESANDIMAGESFOLDER_PATH" -scenenetscripts_path "$SCENENET_SCRIPTS_PATH" -dataset_name ${DATASET_NAME} -dataset_path "$SCENENN_DATASET_PATH" -frame_step ${FRAME_STEP} -end_frame ${END_FRAME}
fi
echo -e "\n**** Pickling files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
# Check whether the pickle files exist already.
if [ -f "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/pickled_train.pkl ] ||
   [ -f "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/pickled_test.pkl ] ||
   [ -f "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/pickled_val.pkl ]
then
  echo -e "Pickle files were already found in the output directory "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/. Please delete or rename them, and relaunch this script to generate new ones."
  exit 1
else
  python "$PYTHONSCRIPTS_PATH"/pickle_files.py -splittingfiles_path "$LINESANDIMAGESFOLDER_PATH" -output_path "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/ -dataset_name ${DATASET_NAME}
fi
for word in test train val all_lines; do
  # NOTE: These files include absolute paths w.r.t to the machine where the
  # data was generated. Still, they are included as a reference to how pickled
  # files were formed (i.e., how data was divided before pickling the data).
  # Further conversion is needed when using these files to train the NN, to
  # replace the absolute path.
  mv "$LINESANDIMAGESFOLDER_PATH"/${word}.txt "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/;
  mv "$LINESANDIMAGESFOLDER_PATH"/${word}_with_line_endpoints.txt "$PICKLEANDSPLIT_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}/;
done

# Delete lines files and virtual camera images.
#echo -e "\n**** Delete lines files for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
#rm "$LINESANDIMAGESFOLDER_PATH"/VALID_LINES_FILES_${TRAJ_NUM}_${DATASET_NAME};
#rm -r "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM};
#echo -e "\n**** Delete virtual camera images for trajectory ${TRAJ_NUM} in ${DATASET_NAME} set ****\n";
#rm "$LINESANDIMAGESFOLDER_PATH"/VALID_VIRTUAL_CAMERA_IMAGES_${TRAJ_NUM}_${DATASET_NAME};
#rm -r "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM};
#rm -r "$LINESANDIMAGESFOLDER_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}_inpaint;
