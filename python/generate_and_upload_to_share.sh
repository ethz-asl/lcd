#!/bin/bash
NUM_OF_TRAJ=${1:-5};
SHARE_PATH=${2:-/media/line_tools/fmilano/data/};
for i in $(seq 1 $NUM_OF_TRAJ); do
   # Generate trajectory and move data to share path
   ./generate_trajectory_files.sh $i $SHARE_PATH;
done
# Remove previously existent split files if any
for i in train test val; do
  if [ -e $i.txt ]
  then
    sudo rm $i.txt
  fi
done
# Split dataset
./split_all_trajectories.sh $NUM_OF_TRAJ $SHARE_PATH;
