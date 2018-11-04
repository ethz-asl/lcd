echo "********OLD SCRIPT!!! TODO: FIX********"
NUM_OF_TRAJ=$1;
LINE_PATH=${2:-'/media/line_tools/fmilano/data/'}
for i in $(seq 1 $NUM_OF_TRAJ); do
	python split_dataset_with_labels_world.py -trajectory ${i} -path_to_lines_root ${LINE_PATH}'train_lines/' -path_to_lines_image ${LINE_PATH}'train/';
	echo "Split data of trajectory $i";
done
