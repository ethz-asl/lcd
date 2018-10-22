#!/usr/bin/env bash
# Get trajectory number as first argument,
# 1 by default
TRAJ_NUM=${1:-1}
cd ../data
mkdir -p train_lines/traj_${TRAJ_NUM}
mkdir -p train/traj_${TRAJ_NUM} train/traj_${TRAJ_NUM}_inpaint
cd train/traj_${TRAJ_NUM}
frame="frame_";
rgb="rgb";
depth="depth";
for i in {0..299..1}; do
    mkdir -p "${frame}${i}/${rgb}"
    mkdir -p "${frame}${i}/${depth}"
done
