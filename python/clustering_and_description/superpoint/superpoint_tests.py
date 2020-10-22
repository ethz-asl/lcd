import sys
import os
import cv2
import pickle
import numpy as np
import sklearn.cluster as sc

sys.path.append("../../../../SuperPointPretrainedNetwork/")
import demo_superpoint

RANDOM_LIGHTING = False

map_path = "/nvme/line_nyu/all_data_4_or_more"
vocabulary_path = "superpoint_vocabulary_diml"

# data_path = "/nvme/datasets/interiornet"
# pickle_path = "interiornet_descriptors"

map_path = "/nvme/line_ws/all_data_diml"
data_path = "/nvme/datasets/diml_depth/HD7"
pickle_path = "diml_descriptors"

# data_path = "/nvme/datasets/nyu_v2/HD7"
# pickle_path = "nyu_descriptors"

VOCABULARY_DIM = 1024

if RANDOM_LIGHTING:
    pickle_path = pickle_path + "_random"

map_list = [name for name in os.listdir(map_path)
            if os.path.isdir(os.path.join(map_path, name))]

files_list = [os.path.join(data_path, name) for name in os.listdir(data_path)
              if os.path.isdir(os.path.join(data_path, name)) and name in map_list]

# Superpoint options:
weights_path = "/home/felix/line_ws/src/SuperPointPretrainedNetwork/superpoint_v1.pth"

fe = demo_superpoint.SuperPointFrontend(weights_path=weights_path,
                                        nms_dist=4,
                                        conf_thresh=0.015,
                                        nn_thresh=0.7,
                                        cuda=True)

if not os.path.isfile(vocabulary_path):
    print("Computing Superpoint descriptor vocabulary.")

    descriptors = []

    for i, scene_dir in enumerate(files_list[:100]):
        if RANDOM_LIGHTING:
            rgb_dir = os.path.join(scene_dir, "random_lighting_cam0", "data")
        else:
            rgb_dir = os.path.join(scene_dir, "cam0", "data")

        print("{}% completed; computing scene {}".format(i * 1.0 / len(files_list[:100]) * 100., scene_dir))

        frame_list = [os.path.join(rgb_dir, name) for name in os.listdir(rgb_dir)
                      if os.path.isfile(os.path.join(rgb_dir, name))]

        for frame_path in frame_list:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype('float32') / 255.

            pts, desc, heatmap = fe.run(frame)

            if desc is not None:
                assert(desc.shape[0] == 256)
                descriptors.append(np.transpose(desc))

    all_embeddings = np.vstack(descriptors)
    print("Starting k means clustering.")
    kmeans = sc.MiniBatchKMeans(n_clusters=VOCABULARY_DIM)
    kmeans.fit(all_embeddings)
    print("Finshed clustering")
    with open(vocabulary_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print("Saved clustering file sift_vocabulary")
else:
    print("Loading Superpoint vocabulary.")
    with open(vocabulary_path, 'rb') as f:
        kmeans = pickle.load(f)


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
        frame = frame.astype('float32') / 255.

        pts, desc, heatmap = fe.run(frame)

        if desc is not None:
            indices = kmeans.predict(np.transpose(desc))
            superpoint_embedding = np.histogram(indices, np.arange(VOCABULARY_DIM + 1))[0].astype(float)

            results[frame_path] = superpoint_embedding
        else:
            results[frame_path] = np.zeros((VOCABULARY_DIM, ))

with open(pickle_path, 'wb') as f:
    pickle.dump(results, f)
