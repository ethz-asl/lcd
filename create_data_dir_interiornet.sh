#!/usr/bin/env bash
# Get trajectory number as first argument,
# 1 by default.
TRAJ_NUM=${1:-1}
DATA_PATH=${2:-"../../data"}
DATASET_NAME=${3:-"interiornet"}
START_FRAME=${4:-0}
END_FRAME=${5:-300}
FRAME_STEP=${6:-1}

echo "DATA_PATH is $DATA_PATH"

mkdir -p "$DATA_PATH"/${DATASET_NAME}_lines/traj_${TRAJ_NUM}
mkdir -p "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM} "$DATA_PATH"/${DATASET_NAME}/traj_${TRAJ_NUM}_inpaint

frame="frame_";
rgb="rgb";
depth="depth";
for i in $(seq $START_FRAME $FRAME_STEP $END_FRAME); do
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${rgb}"
    mkdir -p "$DATA_PATH/${DATASET_NAME}/traj_${TRAJ_NUM}/${frame}${i}/${depth}"
done
