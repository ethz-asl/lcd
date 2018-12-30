#!/usr/bin/env bash
# Get trajectory number as first argument,
# 1 by default
TRAJ_NUM=${1:-1}
DATA_PATH=${2:-"../data"}
DATASET_NAME=${3:-"train_0"}

# Check that name of the dataset is valid
case ${DATASET_NAME%_*} in
    train)
        if [ ${DATASET_NAME#*_} -ge 0 ] && [ ${DATASET_NAME#*_} -le 16 ]
        then
          echo "Using training set ${DATASET_NAME#*_} from pySceneNetRGBD."
        else
          echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16."
          exit 1
        fi
        ;;
    val)
        echo "Using validation set from pySceneNetRGBD."
        ;;
    *)
        echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16."
        exit 1
        ;;
esac

echo "DATA_PATH is $DATA_PATH"

mkdir -p "$DATA_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}
mkdir -p "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM} "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}_inpaint

frame="frame_";
rgb="rgb";
depth="depth";
for i in {0..299..1}; do
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${rgb}"
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${depth}"
done
