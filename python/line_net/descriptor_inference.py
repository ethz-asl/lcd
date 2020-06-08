import numpy as np
import sklearn.neighbors as sn
import sklearn.cluster as sc
import datagenerator_framewise
import visualize_lines
import model
import pickle
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from collections import Counter


COLORS = (visualize_lines.get_colors() * 255).astype(int)
COLORS = np.vstack([np.array([
    [255, 0, 0],
    [128, 64, 64],
    [255, 128, 0],
    [255, 255, 0],
    [255, 128, 0],
    [0, 128, 0],
    [0, 255, 64],
    [0, 255, 255],
    [0, 0, 255],
    [128, 128, 255],
    [128, 128, 0],
    [255, 0, 255],
    [64, 128, 128],
    [128, 128, 64],
    [128, 64, 128],
    [128, 0, 255],
]), COLORS])
INTERIORNET_PATH = "/nvme/datasets/interiornet"


class QueryFloor:
    def __init__(self, scenes):
        self.scenes = scenes


class QueryScene:
    def __init__(self, frames):
        self.frames = frames


class QueryFrame:
    def __init__(self, geometry, labels, path, clusters, gt_clusters):
        self.geometry = geometry
        self.labels = labels
        self.path = path
        self.clusters = clusters
        self.gt_clusters = gt_clusters

    def draw_frame(self, cluster_id=None):
        scene_name = self.path.split('/')[-2]
        frame_id = self.path.split('_')[-1]
        path = os.path.join(INTERIORNET_PATH, scene_name, 'cam0/data', frame_id + '.png')
        # print("Loading image from path " + path)

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        for i, cluster in enumerate(self.clusters):
            if cluster_id is not None and i != cluster_id:
                continue
            color = (int(COLORS[cluster.label, 0]),
                     int(COLORS[cluster.label, 1]),
                     int(COLORS[cluster.label, 2]), 255)
            image = draw_cluster(image, cluster.geometry, color)

        return image


class QueryCluster:
    def __init__(self, geometry, embedding, label):
        self.geometry = geometry
        self.embedding = embedding
        self.label = label


def unnormalize(line_geometries):
    mean = np.array([0., 0., 3.])
    line_geometries[:, :6] = line_geometries[:, :6] * 2.
    line_geometries[:, :3] = line_geometries[:, :3] + mean
    line_geometries[:, 3:6] = line_geometries[:, 3:6] + mean

    return line_geometries


def camera_intrinsic_transform(fx=600,
                               fy=600,
                               pixel_width=640,
                               pixel_height=480):
    """ Gets camera intrinsics matrix for InteriorNet dataset.
    """
    camera_intrinsics = np.zeros((4, 4))
    camera_intrinsics[2, 2] = 1.
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[1, 2] = pixel_height / 2.0
    camera_intrinsics[3, 3] = 1.

    return camera_intrinsics


def draw_cluster(image, line_geometries, color):
    t_cam = camera_intrinsic_transform()
    start_points = line_geometries[:, :3]
    start_points = np.vstack([start_points.T, np.ones((start_points.shape[0],))])
    start_points = t_cam.dot(start_points).T
    start_points = np.rint(start_points / np.expand_dims(start_points[:, 2], -1)).astype(int)[:, :2]
    end_points = line_geometries[:, 3:6]
    end_points = np.vstack([end_points.T, np.ones((end_points.shape[0],))])
    end_points = t_cam.dot(end_points).T
    end_points = np.rint(end_points / np.expand_dims(end_points[:, 2], -1)).astype(int)[:, :2]

    out_image = image
    for i in range(start_points.shape[0]):
        out_image = cv2.line(out_image,
                             (start_points[i, 0], start_points[i, 1]),
                             (end_points[i, 0], end_points[i, 1]),
                             color, 3)

    return out_image


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


def get_sift_vocabulary(interiornet_path, map_data, vocabulary_dim=768, num_scenes=None):
    scene_paths = [os.path.join(interiornet_path, scene.name) for scene in map_data.scenes]

    sift = cv2.xfeatures2d.SIFT_create()

    if not os.path.isfile("sift_vocabulary"):
        all_embeddings = []

        random_scene_paths = scene_paths.copy()
        np.random.shuffle(random_scene_paths)
        for i, scene_path in enumerate(random_scene_paths):
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
        print("Start clustering.")
        kmeans = sc.MiniBatchKMeans(n_clusters=vocabulary_dim)
        kmeans.fit(all_embeddings)
        print("Finshed clustering")
        with open("sift_vocabulary", 'wb') as f:
            pickle.dump(kmeans, f)
        print("Saved clustering file sift_vocabulary")
    else:
        print("Loading sift vocabulary.")
        with open("sift_vocabulary", 'rb') as f:
            kmeans = pickle.load(f)

    return kmeans


def get_sift_embeddings(interiornet_path, map_data, num_scenes=None):
    scene_paths = [os.path.join(interiornet_path, scene.name) for scene in map_data.scenes]
    floor_ids = map_data.floor_ids

    sift = cv2.xfeatures2d.SIFT_create()

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

            if descriptors is not None:
                indices = kmeans.predict(descriptors)
                embedding = np.histogram(indices, np.arange(vocabulary_dim))[0].astype(float)
                descriptors = np.expand_dims(embedding, 0)

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

    return query_frame_embeddings, query_scene_ids, None, map_embeddings, map_scene_ids, None


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
                            scene.get_cluster(i, map_data.max_cluster_line_count,
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
    valid_frame_num = 0
    total_cluster_count = 0
    map_scene_ids = np.array(map_scene_ids)

    for i, embeddings in enumerate(query_frame_embeddings):
        if len(embeddings) > min_num_clusters:
            query_scene_id = query_scene_ids[i]
            matches = []
            print("==========")
            for embedding in embeddings:
                dist, ind = map_tree.query(np.array(embedding).reshape(-1, 1).T, k=k)
                # print("Distances: ")
                # print(dist)
                new_matches = map_scene_ids[np.array(ind[0])]
                # for match in ind[0]:
                #     print(map_scene_names[match])
                # print("Matches: ")
                # print(new_matches)
                matches += new_matches.tolist()
            most_common_matches = Counter(matches).most_common(10)
            # print("Query scene id: {}".format(query_scene_id))
            # print("Query scene name: {}".format(query_scene_names[i]))
            # print("Most common matches: {}".format(most_common_matches))
            if most_common_matches[0][0] == query_scene_id:
                correctly_matched_num += 1
                for embedding in embeddings:
                    dist, ind = map_tree.query(np.array(embedding).reshape(-1, 1).T, k=k)
                    print("Distances: ")
                    print(dist)
                    new_matches = map_scene_ids[np.array(ind[0])]
                    print("Matches: ")
                    print(new_matches)
                    for match in ind[0]:
                        print(map_scene_names[match])
                print("NICE, correctly matched.")
                print("Query scene name: {}".format(query_scene_names[i]))
            total_cluster_count += len(embeddings)
            valid_frame_num += 1
        total_frame_num += 1

    print("Ratio of correctly matched frames is {}".format(correctly_matched_num / total_frame_num))
    print("Number of frames queried: {}".format(total_frame_num))
    print("Ratio of successfully queried frames: {}".format(correctly_matched_num / 397. / 2.))
    print("Average number of clusters per frame: {}".format(total_cluster_count / total_frame_num))


def query_on_floors(query_floors, k=10):
    num_floors = 100

    sift_data = []
    gt_data = []
    data = []

    # Get fully predicted, gt clustered and sift embeddings.

    embeddings = []
    gt_embeddings = []
    sift_embeddings = []

    for floor in query_floors[:num_floors]:
        for scene in floor.scenes:
            for frame in scene.frames:
                for cluster in frame.clusters:
                    embeddings.append(cluster.embedding)
                    data.append((floor, scene, frame, cluster))
                for gt_cluster in frame.gt_clusters:
                    gt_embeddings.append(gt_cluster.embedding)
                    gt_data.append((floor, scene, frame, gt_cluster))
                sift_embeddings.append(frame.sift_embedding)
                sift_data.append((floor, scene, frame))

    embeddings = np.vstack(embeddings)
    gt_embeddings = np.vstack(gt_embeddings)
    sift_embeddings = np.vstack(sift_embeddings)
    # Generating map trees.
    print("Generating map trees.")
    map_tree = sn.KDTree(embeddings, leaf_size=10)
    gt_tree = sn.KDTree(gt_embeddings, leaf_size=10)
    sift_tree = sn.KDTree(sift_embeddings, leaf_size=10)

    print("Starting query.")

    matched_num_clusters = 0
    tot_num_clusters = 0
    matched_num_frames = 0
    empty_num_frames = 0
    tot_num_frames = 0

    tot_num_gt_clusters = 0
    matched_num_gt_clusters = 0
    matched_num_gt_frames = 0

    num_sift_frames = 0
    matched_num_sift_frames = 0
    for floor in query_floors[:num_floors]:
        for scene in floor.scenes:
            for frame in scene.frames:
                # Query fully predicted clusters.

                frame_matches = []
                for i, cluster in enumerate(frame.clusters):
                    dist, nearest = map_tree.query(cluster.embedding.reshape(-1, 1).T, k=k+1)
                    # Warning, the nearest one is of course the very same cluster.
                    nearest_data = [data[nearest[0][j]] for j in range(1, k+1)]
                    matched = False
                    cluster_matches = []
                    for data_point in nearest_data:
                        if data_point[0] is floor:
                            matched = True
                        cluster_matches.append(query_floors.index(data_point[0]))
                    if matched:
                        matched_num_clusters += 1

                    frame_matches += cluster_matches
                    tot_num_clusters += 1
                    # image_1 = frame.draw_frame(i)
                    # image_2 = nearest_data[0][2].draw_frame(nearest_data[0][2].clusters.index(nearest_data[0][3]))
                    # plt.imshow(np.concatenate([image_1, image_2], axis=1))
                    # plt.show()

                if len(frame_matches) > 0:
                    most_common_matches = Counter(frame_matches).most_common(1)
                    if query_floors[most_common_matches[0][0]] is floor:
                        # print("Match.")
                        matched_num_frames += 1
                    else:
                        ...
                        # print("...")
                else:
                    print("Empty frame.")
                    empty_num_frames += 1

                # Query gt clusters.

                gt_frame_matches = []
                for i, gt_cluster in enumerate(frame.gt_clusters):
                    dist, nearest = gt_tree.query(gt_cluster.embedding.reshape(-1, 1).T, k=k + 1)
                    # Warning, the nearest one is of course the very same cluster.
                    nearest_data = [gt_data[nearest[0][j]] for j in range(1, k + 1)]
                    matched = False
                    cluster_matches = []
                    for data_point in nearest_data:
                        if data_point[0] is floor:
                            matched = True
                        cluster_matches.append(query_floors.index(data_point[0]))
                    if matched:
                        matched_num_gt_clusters += 1

                    gt_frame_matches += cluster_matches
                    tot_num_gt_clusters += 1

                if len(gt_frame_matches) > 0:
                    most_common_matches = Counter(gt_frame_matches).most_common(1)
                    if query_floors[most_common_matches[0][0]] is floor:
                        matched_num_gt_frames += 1

                # Query sift embeddings.

                if frame.sift_embedding is not None:
                    dist, ind = sift_tree.query(frame.sift_embedding.reshape(-1, 1).T, k=2)
                    if sift_data[ind[0][1]][0] is floor:
                        matched_num_sift_frames += 1
                    num_sift_frames += 1

                tot_num_frames += 1

    print("==================================================")
    print("Fully predicted:")
    print(" ")
    print("Correctly matched clusters: {}/{}".format(matched_num_clusters, tot_num_clusters))
    print("Ratio of correctly matched clusters: {}".format(matched_num_clusters / tot_num_clusters))
    print(" ")
    print("Correctly matched frames: {}/{}".format(matched_num_frames, tot_num_frames))
    # print("Number of empty frames: {}".format(empty_num_frames))
    print("Ratio of correctly matched frames: {}".format(matched_num_frames / tot_num_frames))
    print("==================================================")
    print("Predictions with ground truth clusters:")
    print(" ")
    print("Correctly matched clusters: {}/{}".format(matched_num_gt_clusters, tot_num_gt_clusters))
    print("Ratio of correctly matched clusters: {}".format(matched_num_gt_clusters / tot_num_gt_clusters))
    print(" ")
    print("Correctly matched frames: {}/{}".format(matched_num_gt_frames, tot_num_frames))
    print("Ratio of correctly matched frames: {}".format(matched_num_gt_frames / tot_num_frames))
    print("==================================================")
    print("Predictions sift image similarity:")
    print(" ")
    print("Correctly matched frames: {}/{}".format(matched_num_sift_frames, tot_num_frames))
    print("Ratio of correctly matched frames: {}".format(matched_num_sift_frames / tot_num_frames))
    print("Number of sift frames: {}".format(num_sift_frames))
    print("==================================================")


def get_full_embeddings(cluster_model, descriptor_model, frame_data,
                        min_cluster_line_count=4,
                        max_cluster_line_count=50):
    scenes = frame_data.scenes
    floor_ids = get_floor_ids(scenes)
    training_plan = frame_data.training_plan

    vocabulary_dim = 768
    kmeans = get_sift_vocabulary(INTERIORNET_PATH, frame_data, 768)
    sift = cv2.xfeatures2d.SIFT_create()

    query_floors = [QueryFloor([]) for i in range(np.unique(floor_ids).shape[0])]
    query_scenes = [QueryScene([]) for i in range(len(scenes))]

    for idx, element in enumerate(training_plan):
        scene_id = element[0]
        floor_id = floor_ids[scene_id]
        frame_id = element[1]
        frame_path = scenes[scene_id].frame_paths[frame_id]

        # Get predicted embeddings using full prediction.

        data, _ = frame_data.__getitem__(idx)
        frame_geometries = data['lines']
        frame_images = data['images']
        valid_mask = data['valid_input_mask']
        gt_labels = data['labels']

        cluster_output = cluster_model.predict_on_batch(data)
        cluster_output = cluster_output[0, valid_mask[0, :], :]
        cluster_output = np.argmax(cluster_output, axis=-1)

        query_frame = QueryFrame(frame_geometries[0, valid_mask[0, :], :], cluster_output, frame_path, [], [])

        unique_output = np.unique(cluster_output)
        # unique_output = unique_output[np.where(unique_output != 0)]
        cluster_embeddings = []
        for label_idx in unique_output:
            # assert(label_idx != 0)
            line_indices = np.where(cluster_output == label_idx)[0][:max_cluster_line_count]
            cluster_line_count = len(line_indices)
            # Min cluster line count is 4.
            if cluster_line_count >= min_cluster_line_count:
                cluster_lines = frame_geometries[0, line_indices[:cluster_line_count], :]
                cluster_lines_render = unnormalize(cluster_lines.copy())
                cluster_lines = datagenerator_framewise.set_mean_zero(cluster_lines)

                cluster_valid_mask = np.zeros((1, max_cluster_line_count), dtype=bool)
                cluster_geometries = np.zeros((1, max_cluster_line_count, frame_geometries.shape[2]))
                cluster_images = np.zeros((1, max_cluster_line_count,
                                           frame_images.shape[2], frame_images.shape[3], frame_images.shape[4]))
                cluster_valid_mask[:, :cluster_line_count] = True
                cluster_geometries[0:, :cluster_line_count, :] = cluster_lines
                cluster_images[:, :cluster_line_count, :, :, :] = \
                    frame_images[:, line_indices[:cluster_line_count], :, :, :]

                cluster_data = {
                    'lines': cluster_geometries,
                    'valid_input_mask': cluster_valid_mask,
                    'images': cluster_images,
                    'ones_model': np.expand_dims(np.ones((1, 1)), axis=0)
                }
                cluster_embedding = descriptor_model.predict_on_batch(cluster_data)
                cluster_embedding = np.array(cluster_embedding)[0, :]
                cluster_embeddings.append(cluster_embedding)

                query_frame.clusters.append(QueryCluster(cluster_lines_render, cluster_embedding, label_idx))

        # Get embeddings from ground truth clustering.

        gt_labels = gt_labels[0, valid_mask[0, :]]
        unique_gt = np.unique(gt_labels)
        gt_embeddings = []
        for j, label_idx in enumerate(unique_gt):
            line_indices = np.where(gt_labels == label_idx)[0][:max_cluster_line_count]
            cluster_line_count = len(line_indices)
            # Min cluster line count is 4.
            if cluster_line_count >= min_cluster_line_count:
                cluster_lines = frame_geometries[0, line_indices[:cluster_line_count], :]
                cluster_lines_render = unnormalize(cluster_lines.copy())
                cluster_lines = datagenerator_framewise.set_mean_zero(cluster_lines)

                cluster_valid_mask = np.zeros((1, max_cluster_line_count), dtype=bool)
                cluster_geometries = np.zeros((1, max_cluster_line_count, frame_geometries.shape[2]))
                cluster_images = np.zeros((1, max_cluster_line_count,
                                           frame_images.shape[2], frame_images.shape[3], frame_images.shape[4]))
                cluster_valid_mask[:, :cluster_line_count] = True
                cluster_geometries[0:, :cluster_line_count, :] = cluster_lines
                cluster_images[:, :cluster_line_count, :, :, :] = \
                    frame_images[:, line_indices[:cluster_line_count], :, :, :]

                cluster_data = {
                    'lines': cluster_geometries,
                    'valid_input_mask': cluster_valid_mask,
                    'images': cluster_images,
                    'ones_model': np.expand_dims(np.ones((1, 1)), axis=0)
                }
                gt_embedding = descriptor_model.predict_on_batch(cluster_data)
                gt_embedding = np.array(gt_embedding)[0, :]
                gt_embeddings.append(gt_embedding)

                query_frame.gt_clusters.append(QueryCluster(cluster_lines_render, gt_embedding, j))

        # Get SIFT embeddings.

        scene_name = frame_path.split('/')[-2]
        frame_id = frame_path.split('_')[-1]
        image_path = os.path.join(INTERIORNET_PATH, scene_name, 'cam0/data', frame_id + '.png')
        frame_im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(frame_im, None)
        if descriptors is not None:
            indices = kmeans.predict(descriptors)
            sift_embedding = np.histogram(indices, np.arange(vocabulary_dim))[0].astype(float)
            query_frame.sift_embedding = sift_embedding
        else:
            query_frame.sift_embedding = None

        # Add everything to the list.

        query_scenes[scene_id].frames.append(query_frame)
        if query_scenes[scene_id] not in query_floors[floor_id].scenes:
            query_floors[floor_id].scenes.append(query_scenes[scene_id])

        print("Scene {}, frame {}, number of clusters: {}, gt clusters: {}".format(scene_id,
                                                                                   frame_id,
                                                                                   len(cluster_embeddings),
                                                                                   len(gt_embeddings)))

    return query_floors


if __name__ == '__main__':

    map_dir = "/nvme/line_ws/val_map"
    model_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/description_040620_1846/weights_only.20.hdf5"
    line_model_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/cluster_060620_0111/weights_only.18.hdf5"
    pickle_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/description_040620_1846"
    line_num_attr = 15
    img_shape = (64, 96, 3)
    max_cluster_line_count = 70
    min_cluster_line_count = 4
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
            if query_mode == 'full':
                query_floors = pickle.load(f)
            else:
                (query_frame_embeddings, query_scene_ids, query_scene_names,
                 map_embeddings, map_scene_ids, map_scene_names) = pickle.load(f)
    else:
        print("Loading data.")
        if query_mode == 'only_clusters' or query_mode == 'sift':
            map_data = datagenerator_framewise.ClusterDataSequence(map_dir, batch_size, bg_classes, valid_classes,
                                                                   shuffle=False,
                                                                   data_augmentation=False,
                                                                   img_shape=img_shape,
                                                                   min_line_count=min_cluster_line_count,
                                                                   max_line_count=max_cluster_line_count,
                                                                   load_images=True,
                                                                   training_mode=False)

        if query_mode == 'only_clusters':
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

            model = model.load_cluster_embedding_model(model_path, line_num_attr, max_cluster_line_count,
                                                       embedding_dim, img_shape, margin)

            print("Computing embeddings.")
            query_frame_embeddings, query_scene_ids, query_scene_names, \
                map_embeddings, map_scene_ids, map_scene_names = get_embeddings(model, map_data)
            with open(pickle_path, 'wb') as f:
                pickle.dump((query_frame_embeddings, query_scene_ids, query_scene_names,
                             map_embeddings, map_scene_ids, map_scene_names), f)
        elif query_mode == 'sift':
            interiornet_path = "/nvme/datasets/interiornet"

            query_frame_embeddings, query_scene_ids, query_scene_names, \
                map_embeddings, map_scene_ids, map_scene_names = \
                    get_sift_embeddings(interiornet_path, map_data)
            with open(pickle_path, 'wb') as f:
                pickle.dump((query_frame_embeddings, query_scene_ids, query_scene_names,
                             map_embeddings, map_scene_ids, map_scene_names), f)
        elif query_mode == 'full':
            print("Using end to end clustering.")

            max_frame_line_count = 220
            min_frame_line_count = 20
            max_clusters = 15

            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            line_model = model.load_line_net_model(line_model_path, line_num_attr, max_frame_line_count,
                                                   max_clusters, img_shape)
            embedding_model = model.load_cluster_embedding_model(model_path, line_num_attr, max_cluster_line_count,
                                                                 embedding_dim, img_shape, margin)

            frame_data = datagenerator_framewise.LineDataSequence(map_dir,
                                                                  batch_size,
                                                                  bg_classes,
                                                                  fuse=False,
                                                                  data_augmentation=False,
                                                                  img_shape=img_shape,
                                                                  min_line_count=min_frame_line_count,
                                                                  max_line_count=max_frame_line_count,
                                                                  max_cluster_count=max_clusters,
                                                                  training_mode=False)

            query_floors = get_full_embeddings(line_model, embedding_model, frame_data,
                                               min_cluster_line_count=min_cluster_line_count,
                                               max_cluster_line_count=max_cluster_line_count)

            with open(pickle_path, 'wb') as f:
                pickle.dump(query_floors, f)
        else:
            print("ERROR, query_mode not valid.")
            exit()

    print("Starting query.")
    if query_mode == 'only_clusters':
        query_on_frames(query_frame_embeddings, query_scene_ids, query_scene_names, map_embeddings, map_scene_ids,
                        map_scene_names, k=10, min_num_clusters=1)
    elif query_mode == 'full':
        query_on_floors(query_floors, k=8)
        exit()
        for scene in query_floors[0].scenes:
            for frame in scene.frames:
                img = frame.draw_frame()
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.show()
    elif query_mode == 'sift':
        query_on_sift_frames(query_frame_embeddings, query_scene_ids, map_embeddings, map_scene_ids, k=1)

    # query_data = datagenerator_framewise.ClusterDataSequence(query_dir, batch_size, bg_classes, valid_classes,
    #                                                          shuffle=False, data_augmentation=False,
    #                                                          img_shape=img_shape, max_line_count=50,
    #                                                          load_images=True, training_mode=False)



