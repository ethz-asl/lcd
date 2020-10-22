import sys
import os
import cv2

sys.path.append("../../../SuperPointPretrainedNetwork/")

RANDOM_LIGHTING = True

data_path = "/nvme/datasets/interiornet"
pickle_path = "superpoint/interiornet_descriptors"

if RANDOM_LIGHTING:
    pickle_path = pickle_path + "_random"

files_list = [os.path.join(data_path, name) for name in os.listdir(data_path)
              if os.path.isdir(os.path.join(data_path, name))]

# Superpoint options:
weights_path = "/home/felix/line_ws/src/SuperPointPretrainedNetwork/superpoint_v1.pth"

fe = demo_superpoint.SuperPointFrontend(weights_path=weights_path,
                                        nms_dist=4,
                                        conf_thresh=0.015,
                                        nn_thresh=0.7,
                                        cuda=True)

print("Computing Superpoint + BoW descriptors.")

results = {}

for i, scene_dir in enumerate(files_list):
    if RANDOM_LIGHTING:
        rgb_dir = os.path.join(scene_dir, "random_lighting_cam0", "data")
    else:
        rgb_dir = os.path.join(scene_dir, "cam0", "data")

    print("{}% completed; computing scene {}".format(i * 1.0 / len(files_list) * 100., scene_dir))

    frame_list = [os.path.join(rgb_dir, name) for name in os.listdir(rgb_dir)
                  if os.path.isfile(os.path.join(rgb_dir, name))]

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pts, desc, heatmap = fe.run(frame)

        results[frame_path] = result

with open(pickle_path, 'wb') as f:
    pickle.dump(results, f)
