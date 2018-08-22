#!/usr/bin/env bash

cd ../data
mkdir -p train/traj_1 train/traj_1_inpaint
cd train/traj_1
frame="frame_"
rgb="rgb"
depth="depth"
for i in {0..299..1}; do
    mkdir -p "${frame}${i}/${rgb}"
    mkdir -p "${frame}${i}/${depth}"
done
