#!/usr/bin/env bash
# Get trajectory number as first argument,
# 1 by default
TRAJ_NUM=${1:-1}
DATA_PATH=${2:-"../data"}
DATASET_TYPE=${3:-"train"}

# Check that type is valid
case "$DATASET_TYPE" in
    train)
        echo "Using training set from pySceneNetRGBD."
        ;;
    val)
        echo "Using validation set from pySceneNetRGBD."
        ;;
    *)
        echo "Invalid argument $DATASET_TYPE. Valid options are 'train' and 'val'."
        exit 1
        ;;
esac

echo "DATA_PATH is $DATA_PATH"

mkdir -p "$DATA_PATH"/${DATASET_TYPE}_lines/traj_${TRAJ_NUM}
mkdir -p "$DATA_PATH"/${DATASET_TYPE}/traj_${TRAJ_NUM} "$DATA_PATH"/${DATASET_TYPE}/traj_${TRAJ_NUM}_inpaint

frame="frame_";
rgb="rgb";
depth="depth";
for i in {0..299..1}; do
    mkdir -p "$DATA_PATH/${DATASET_TYPE}/traj_${TRAJ_NUM}/${frame}${i}/${rgb}"
    mkdir -p "$DATA_PATH/${DATASET_TYPE}/traj_${TRAJ_NUM}/${frame}${i}/${depth}"
done
