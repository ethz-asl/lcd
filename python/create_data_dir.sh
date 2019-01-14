#!/usr/bin/env bash
# Get trajectory number as first argument,
# 1 by default.
TRAJ_NUM=${1:-1}
DATA_PATH=${2:-"../data"}
DATASET_NAME=${3:-"train_0"}
START_FRAME=${4:-0}
END_FRAME=${5:-300}

# Check that name of the dataset is valid
case ${DATASET_NAME%_*} in
    train)
        if [ ${DATASET_NAME#*_} -ge 0 ] && [ ${DATASET_NAME#*_} -le 16 ]
        then
          echo "Using training set ${DATASET_NAME#*_} from SceneNetRGBD."
        else
          echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16, and 'scenenn'."
          exit 1
        fi
        ;;
    val)
        echo "Using validation set from SceneNetRGBD."
        ;;
    scenenn)
        echo "Using SceneNN dataset."

        ;;
    *)
        echo "Invalid argument $DATASET_NAME. Valid options are 'val' and 'train_NUM', where NUM is between 0 and 16, and 'scenenn'."
        exit 1
        ;;
esac

echo "DATA_PATH is $DATA_PATH"

mkdir -p "$DATA_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}
mkdir -p "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM} "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}_inpaint

frame="frame_";
rgb="rgb";
depth="depth";
for i in $(seq $START_FRAME $END_FRAME); do
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${rgb}"
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${depth}"
done
