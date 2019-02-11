#!/bin/bash
# NOTE: if you move these script please make sure that $LINE_TOOLS_ROOT points
# to the root of line_tools (e.g. ~/catkin_extended_ws/src/line_tools/)
CURRENT_DIR=`dirname $0`
source $CURRENT_DIR/config_paths_and_variables.sh
LINE_TOOLS_ROOT="$CURRENT_DIR"

if [[ `basename $(dirname $(realpath $LINE_TOOLS_ROOT))` != "src" || `basename $(realpath $LINE_TOOLS_ROOT)` != "line_tools" ]];
then
  echo "Please edit/move this script so that LINE_TOOLS_ROOT points to the root of line_tools (e.g. ~/catkin_extended_ws/src/line_tools). Exiting."
  exit 1
fi

# Go to root of line_tools.
cd $LINE_TOOLS_ROOT
if [ -f paths_and_variables.txt ]
then
  rm paths_and_variables.txt;
fi
touch paths_and_variables.txt
echo "SCENENET_DATASET_PATH $SCENENET_DATASET_PATH" >> paths_and_variables.txt
echo "SCENENN_DATASET_PATH $SCENENN_DATASET_PATH" >> paths_and_variables.txt
echo "SCENENET_SCRIPTS_PATH $SCENENET_SCRIPTS_PATH" >> paths_and_variables.txt
echo "BAGFOLDER_PATH $BAGFOLDER_PATH" >> paths_and_variables.txt
echo "LINESANDIMAGESFOLDER_PATH $LINESANDIMAGESFOLDER_PATH" >> paths_and_variables.txt
echo "PYTHONSCRIPTS_PATH $PYTHONSCRIPTS_PATH" >> paths_and_variables.txt
echo "TARFILES_PATH $TARFILES_PATH" >> paths_and_variables.txt
echo "PICKLEANDSPLIT_PATH $PICKLEANDSPLIT_PATH" >> paths_and_variables.txt
echo "TRAJ_NUM $TRAJ_NUM" >> paths_and_variables.txt
echo "DATASET_NAME $DATASET_NAME" >> paths_and_variables.txt
