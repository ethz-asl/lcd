import numpy as np
import sklearn.neighbors as sn
import sklearn.cluster as sc
import datagenerator_framewise
import model
import pickle
import os
import cv2

import tensorflow as tf

from collections import Counter


def get_floor_ids(scenes):
    # The floor names are the name of the floors with multiple scenes, e.g. 3FO4IMXSHSVT
    floor_names = []
    curr_id = 0
    floor_ids = np.zeros((len(scenes),), dtype=int)
    for i, scene in enumerate(scenes):
        floor_name = scene.name.split('_')[0]
        if floor_name in floor_names:
            floor_ids[i] = floor_names.index(floor_name)
        else:
            floor_names.append(floor_name)
            floor_ids[i] = curr_id
            curr_id += 1

    return floor_ids


def get_sift_embeddings(interiornet_path, map_data, num_scenes=None):
    scene_paths = [os.path.join(interiornet_path, scene.name) for scene in map_data.scenes]
    floor_ids = map_data.floor_ids

    sift = cv2.xfeatures2d.SIFT_create()

    all_embeddings = []

    for i, scene_path in enumerate(scene_paths):
        if num_scenes is not None and i > num_scenes:
            print("Stopping with {} frames.".format(num_scenes))
            break

        print("Processing scene {}.".format(i))
        frame_dir = os.path.join(scene_path, "cam0/data")
        frame_paths = [os.path.join(frame_dir, name) for name in os.listdir(frame_dir)
                       if os.path.isfile(os.path.join(frame_dir, name))]

        for j, frame_path in enumerate(frame_paths):
            frame_im = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(frame_im, None)

            if descriptors is not None:
                all_embeddings.append(descriptors)
            else:
                print("Empty frame at " + frame_path)

    print("Starting k means clustering.")
    all_embeddings = np.vstack(all_embeddings)
    print("")
    kmeans = sc.KMeans(n_clusters=256)
    kmeans.fit(all_embeddings)
    print("Finshed clustering")

    query_frame_embeddings = []
    query_scene_ids = []
    map_embeddings = []
    map_scene_ids = []

    query_frames = list(range(2))

    for i, scene_path in enumerate(scene_paths):
        if num_scenes is not None and i > num_scenes:
            print("Stopping with {} frames.".format(num_scenes))
            break

        print("Processing scene {}.".format(i))
        frame_dir = os.path.join(scene_path, "cam0/data")
        frame_paths = [os.path.join(frame_dir, name) for name in os.listdir(frame_dir)
                       if os.path.isfile(os.path.join(frame_dir, name))]
        floor_id = floor_ids[i]

        for j, frame_path in enumerate(frame_paths):
            frame_im = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(frame_im, None)

            indices = kmeans.predict(descriptors)
            print(indices)
            exit()
            if descriptors is not None:
                if j in query_frames:
                    query_frame_embeddings.append(descriptors)
                    query_scene_ids.append(floor_id)
                else:
                    map_embeddings.append(descriptors)
                    map_scene_ids += [floor_id for l in range(descriptors.shape[0])]
            else:
                print("Empty frame at " + frame_path)

    map_embeddings = np.vstack(map_embeddings)
    map_scene_ids = np.array(map_scene_ids)

    return query_frame_embeddings, query_scene_ids, map_embeddings, map_scene_ids


def query_on_sift_frames(query_frame_embeddings, query_scene_ids, map_embeddings, map_scene_ids, k=1):
    map_tree = sn.KDTree(map_embeddings, leaf_size=10)
    correctly_matched_num = 0
    total_frame_num = 0
    total_cluster_count = 0

    for i, embeddings in enumerate(query_frame_embeddings):
        if embeddings.shape[0] > 0:
            query_scene_id = query_scene_ids[i]
            print("==========")
            print(embeddings.shape[0])
            dist, ind = map_tree.query(embeddings[:100, :], k=k)
            matches = map_scene_ids[np.array(ind).flatten()]
            most_common_matches = Counter(matches).most_common(10)
            print("Query scene id: {}".format(query_scene_id))
            print("Most common matches: {}".format(most_common_matches))
            if most_common_matches[0][0] == query_scene_id:
                print("SHIT, correctly matched.")
                correctly_matched_num += 1
            total_cluster_count += len(embeddings)

        total_frame_num += 1

    print("Ratio of correctly matched frames is {}".format(correctly_matched_num / total_frame_num))
    print("Number of frames: {}".format(total_frame_num))
    print("Average number of clusters per frame: {}".format(total_cluster_count / total_frame_num))


def get_embeddings(model, map_data, num_scenes=None, query_frames=list(range(2))):
    scenes = map_data.scenes
    floor_ids = map_data.floor_ids
    max_frame_num = 20
    query_frame_embeddings = []
    query_scene_ids = []
    query_scene_names = []
    map_embeddings = []
    map_scene_ids = []
    map_scene_names = []

    for scene_idx, scene in enumerate(scenes):
        if num_scenes is not None and scene_idx > num_scenes:
            print("Stopping with {} frames.".format(num_scenes))
            break

        print("Starting scene {}.".format(scene_idx))
        for frame_id in range(max_frame_num):
            print("Frame {}.".format(frame_id))
            frame_embeddings = []
            frame_scene_id = floor_ids[scene_idx]

            # frame_clusters = []

            for i in range(len(scene.clusters)):
                for j in range(len(scene.clusters[i])):
                    if scene.clusters[i][j].frame_id == frame_id:
                        line_count, cluster_lines, cluster_label, cluster_class, cluster_images, cluster_id = \
                            scene.get_cluster(i, map_data.max_line_count,
                                              False, center=True, forced_choice=j)

                        geometries, valid_mask, images = map_data.process_cluster(line_count, cluster_lines,
                                                                                  cluster_label,
                                                                                  cluster_class,
                                                                                  cluster_images)

                        data = {
                            'lines': geometries,
                            'valid_input_mask': valid_mask,
                            'images': images,
                            'ones_model': np.expand_dims(np.ones((1, 1)), axis=0)
                        }

                        output = model.predict_on_batch(data)
                        frame_embeddings.append(np.array(output[0, :]))
                        # frame_clusters.append(data)

            # output = model.predict(np.concatenate(frame_clusters, axis=0))

            # for i in range(output.shape[0]):
            #     frame_embeddings.append(output[i, :])

            if len(frame_embeddings) > 0:
                if frame_id in query_frames:
                    query_frame_embeddings.append(frame_embeddings)
                    query_scene_ids.append(frame_scene_id)
                    query_scene_names.append(scene.name + "_frame_{}".format(frame_id))
                else:
                    map_embeddings += frame_embeddings
                    map_scene_ids += [frame_scene_id for l in range(len(frame_embeddings))]
                    map_scene_names += [scene.name + "_frame_{}".format(frame_id) for l in range(len(frame_embeddings))]

    return query_frame_embeddings, query_scene_ids, query_scene_names, map_embeddings, map_scene_ids, map_scene_names


def query_on_frames(query_frame_embeddings, query_scene_ids, query_scene_names,
                    map_embeddings, map_scene_ids, map_scene_names, k=20, min_num_clusters=1):
    print(len(map_embeddings))
    map_embeddings = np.vstack(map_embeddings)
    print("Computing KD tree.")
    map_tree = sn.KDTree(map_embeddings, leaf_size=10)
    print("Finished computing KD tree.")
    correctly_matched_num = 0
    total_frame_num = 0
    total_cluster_count = 0
    map_scene_ids = np.array(map_scene_ids)

    for i, embeddings in enumerate(query_frame_embeddings):
        if len(embeddings) > min_num_clusters:
            query_scene_id = query_scene_ids[i]
            matches = []
            print("==========")
            for embedding in embeddings:
                dist, ind = map_tree.query(np.array(embedding).reshape(-1, 1).T, k=k)
                print("Distances: ")
                print(dist)
                new_matches = map_scene_ids[np.array(ind[0])]
                for match in ind[0]:
                    print(map_scene_names[match])
                # print(map_embeddings[ind[0][0], :])
                # print(embedding)
                # print(map_embeddings.shape[0])
                print("Matches: ")
                print(new_matches)
                matches += new_matches.tolist()
            most_common_matches = Counter(matches).most_common(10)
            print("Query scene id: {}".format(query_scene_id))
            print("Query scene name: {}".format(query_scene_names[i]))
            print("Most common matches: {}".format(most_common_matches))
            if most_common_matches[0][0] == query_scene_id:
                print("NICE, correctly matched.")
                correctly_matched_num += 1
            total_frame_num += 1
            total_cluster_count += len(embeddings)

    print("Ratio of correctly matched frames is {}".format(correctly_matched_num / total_frame_num))
    print("Number of frames: {}".format(total_frame_num))
    print("Average number of clusters per frame: {}".format(total_cluster_count / total_frame_num))


def get_full_embeddings(cluster_model, descriptor_model, frame_data, query_frames=[0, 1]):
    scenes = frame_data.scenes
    floor_ids = get_floor_ids(scenes)
    training_plan = frame_data.training_plan
    query_frame_embeddings = []
    query_scene_ids = []
    query_scene_names = []
    map_embeddings = []
    map_scene_ids = []
    map_scene_names = []

    for idx, element in enumerate(training_plan):
        scene_id = element[0]
        floor_id = floor_ids[scene_id]
        frame_id = element[1]

        data, _ = frame_data.__getitem__(idx)
        frame_geometries = data['lines']
        frame_images = data['images']
        valid_mask = data['valid_input_mask']

        cluster_output = cluster_model.predict_on_batch(data)[0, valid_mask, :]
        cluster_output = np.argmax(cluster_output, axis=-1)
        unique_output = np.unique(cluster_output)
        unique_output = unique_output[np.where(unique_output != 0)]
        cluster_count = 0
        for label_idx in unique_output:
            assert(label_idx != 0)
            line_indices = np.where(cluster_output == label_idx)
            # Min cluster line count is 4.
            if len(line_indices[0]) >= 4:
                cluster_count += 1

        print("Valid number of clusters is: ".format(cluster_count))
        exit()

    return query_frame_embeddings, query_scene_ids, query_scene_names, map_embeddings, map_scene_ids, map_scene_names


if __name__ == '__main__':

    map_dir = "/nvme/line_ws/val_map"
    model_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/description_040620_1846/weights_only.20.hdf5"
    line_model_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/cluster_060620_0111/weights_only.10.hdf5"
    pickle_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/description_040620_1846"
    line_num_attr = 15
    img_shape = (64, 96, 3)
    max_line_count = 50
    min_line_count = 4
    batch_size = 1
    margin = 0.3
    embedding_dim = 256
    bg_classes = [0, 1, 2, 20, 22]
    valid_classes = [i for i in range(41) if i not in bg_classes]

    # only_clusters, sift or full
    query_mode = 'full'

    pickle_name = "map_pickle_" + query_mode
    pickle_path = os.path.join(pickle_path, pickle_name)

    if os.path.isfile(pickle_path):
        print("Found map, loading.")
        with open(pickle_path, 'rb') as f:
            (query_frame_embeddings, query_scene_ids, query_scene_names,
             map_embeddings, map_scene_ids, map_scene_names) = pickle.load(f)
    else:
        print("Loading data.")
        if query_mode == 'only_clusters' or query_mode == 'sift':
            map_data = datagenerator_framewise.ClusterDataSequence(map_dir, batch_size, bg_classes, valid_classes,
                                                                   shuffle=False, data_augmentation=False,
                                                                   img_shape=img_shape, min_line_count=min_line_count,
                                                                   max_line_count=max_line_count,
                                                                   load_images=True, training_mode=False)

        if query_mode == 'only_clusters':
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

            model = model.load_cluster_embedding_model(model_path, line_num_attr, max_line_count,
                                                       embedding_dim, img_shape, margin)

            print("Computing embeddings.")
            query_frame_embeddings, query_scene_ids, query_scene_names, \
                map_embeddings, map_scene_ids, map_scene_names = get_embeddings(model, map_data)
        elif query_mode == 'sift':
            interiornet_path = "/nvme/datasets/interiornet"

            query_frame_embeddings, query_scene_ids, query_scene_names, \
                map_embeddings, map_scene_ids, map_scene_names = \
                    get_sift_embeddings(interiornet_path, map_data, num_scenes=10)
        elif query_mode == 'full':
            print("Using end to end clustering.")

            max_frame_line_count = 220
            min_frame_line_count = 30
            max_clusters = 15

            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            line_model = model.load_line_net_model(line_model_path, line_num_attr, max_frame_line_count,
                                                   max_clusters, img_shape)
            embedding_model = model.load_cluster_embedding_model(model_path, line_num_attr, max_line_count,
                                                                 embedding_dim, img_shape, margin)

            frame_data = datagenerator_framewise.LineDataSequence(map_dir,
                                                                  batch_size,
                                                                  bg_classes,
                                                                  fuse=False,
                                                                  img_shape=img_shape,
                                                                  min_line_count=min_frame_line_count,
                                                                  max_line_count=max_frame_line_count,
                                                                  max_cluster_count=max_clusters,
                                                                  training_mode=False)

            query_frame_embeddings, query_scene_ids, query_scene_names, \
                map_embeddings, map_scene_ids, map_scene_names = \
                    get_full_embeddings(line_model, embedding_model, frame_data)
        with open(pickle_path, 'wb') as f:
            pickle.dump((query_frame_embeddings, query_scene_ids, query_scene_names,
                         map_embeddings, map_scene_ids, map_scene_names), f)

    print("Starting query.")
    if query_mode == 'only_clusters':
        query_on_frames(query_frame_embeddings, query_scene_ids, query_scene_names, map_embeddings, map_scene_ids,
                        map_scene_names, k=1, min_num_clusters=1)
    elif query_mode == 'sift':
        query_on_sift_frames(query_frame_embeddings, query_scene_ids, map_embeddings, map_scene_ids, k=6)

    # query_data = datagenerator_framewise.ClusterDataSequence(query_dir, batch_size, bg_classes, valid_classes,
    #                                                          shuffle=False, data_augmentation=False,
    #                                                          img_shape=img_shape, max_line_count=50,
    #                                                          load_images=True, training_mode=False)



