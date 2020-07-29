""" Perform place recognition experiments with the full pipeline, the full pipeline with ground truth clustering
    and the SIFT bag-of-words approach.
"""
import numpy as np
import sklearn.neighbors as sn
import sklearn.cluster as sc
import sklearn.metrics as sm
import datagenerator_framewise
import visualize_clusters
import model
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import inference_agglomerative

import tensorflow as tf

from collections import Counter

import sys
sys.path.append("../tools/")
import interiornet_utils

# Generate some colors for visualization.
COLORS = (visualize_clusters.get_colors() * 255).astype(int)
COLORS = np.vstack([np.array([
    [255, 0, 0],
    [128, 64, 64],
    [255, 128, 0],
    [255, 255, 0],
    [220, 255, 0],
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


class QueryFloor:
    """
    A data class containing all scenes of a floor. Only relevant for the InteriorNet dataset.
    """
    def __init__(self, scenes):
        self.scenes = scenes


class QueryScene:
    """
    A data class containing all frames of a scene.
    """
    def __init__(self, frames):
        self.frames = frames


class QueryFrame:
    """
    A data class containing the data for a frame. That is the predicted and ground truth clusters, the path
    to the RGB image, the line geometries.
    """
    def __init__(self, geometry, labels, path, clusters, gt_clusters):
        self.geometry = geometry
        self.labels = labels
        self.path = path
        self.clusters = clusters
        self.gt_clusters = gt_clusters

    def draw_frame(self, cluster_id=None, predicted=True):
        """
        Renders the full frame on the corresponding RGB image using colored lines.
        :param cluster_id: If specified, renders only this cluster.
        :param predicted: If True, renders the predicted clusters, if False, renders the ground truth clusters.
        """
        scene_name = self.path.split('/')[-2]
        frame_id = self.path.split('_')[-1]
        path = os.path.join(DATASET_PATH, scene_name, 'cam0/data', frame_id + '.png')

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image[:, :, :3] = (image[:, :, :3] * 0.6).astype(np.uint8)

        if predicted:
            clusters = self.clusters
        else:
            clusters = self.gt_clusters
        for i, cluster in enumerate(clusters):
            if cluster_id is not None and i != cluster_id:
                continue
            color_idx = cluster.label
            color = (int(COLORS[color_idx, 2]),
                     int(COLORS[color_idx, 1]),
                     int(COLORS[color_idx, 0]), 255)
            image = draw_cluster(image, cluster.geometry, color)

        return image


class QueryCluster:
    """
    A data class to hold the data of a cluster, including the geometry data of the lines, the
    descriptor embedding, the semantic label and the number of lines.
    """
    def __init__(self, geometry, embedding, label, line_count):
        self.geometry = geometry
        self.embedding = embedding
        self.label = label
        self.line_count = line_count


def unnormalize(line_geometries):
    """
    Adds the train set mean back to the geometries and multiplies the start and end points by 2.
    :param line_geometries: The geometric data of the lines in a numpy array of shape (N, 15)
    :return: The unnormalized geometry of the lines.
    """
    mean = np.array([0., 0., 3.])
    line_geometries[:, :6] = line_geometries[:, :6] * 2.
    line_geometries[:, :3] = line_geometries[:, :3] + mean
    line_geometries[:, 3:6] = line_geometries[:, 3:6] + mean

    return line_geometries


def draw_cluster(image, line_geometries, color):
    """
    Function to render line clusters in an image.
    :param image: The RGB image to draw the lines to.
    :param line_geometries: The geometric data of the lines in a numpy array of shape (N, 15)
    :param color: The color of the lines.
    """
    t_cam = interiornet_utils.camera_intrinsic_transform()
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
                             color, 2)

    return out_image


def get_floor_ids(scenes):
    """
    Returns the unique floor id of each scene by checking the scene names.
    :param scenes: A list with the scenes of the dataset.
    :return: A numpy of floor ids, with the same length as the scenes list.
    """
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


def get_sift_vocabulary(interiornet_path, sift_vocabulary_path, map_data, vocabulary_dim=768, num_scenes=None):
    scene_paths = [os.path.join(interiornet_path, scene.name) for scene in map_data.scenes]

    sift = cv2.xfeatures2d.SIFT_create()

    # If the sift vocabulary was already computed, use it.
    if not os.path.isfile(sift_vocabulary_path):
        all_embeddings = []

        random_scene_paths = scene_paths.copy()
        np.random.shuffle(random_scene_paths)
        # Obtain SIFT features for all frames in dataset.
        for i, scene_path in enumerate(random_scene_paths):
            if num_scenes is not None and i > num_scenes:
                print("Stopping with {} frames.".format(num_scenes))
                break

            print("Processing SIFT features of scene {}.".format(i))
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

        # Use k-means clustering to compute the bag-of-words vocabulary.
        print("Starting k means clustering.")
        all_embeddings = np.vstack(all_embeddings)
        print("Start clustering.")
        kmeans = sc.MiniBatchKMeans(n_clusters=vocabulary_dim)
        kmeans.fit(all_embeddings)
        print("Finshed clustering")
        with open(sift_vocabulary_path, 'wb') as f:
            pickle.dump(kmeans, f)
        print("Saved clustering file sift_vocabulary")
    else:
        print("Loading sift vocabulary.")
        with open(sift_vocabulary_path, 'rb') as f:
            kmeans = pickle.load(f)

    return kmeans


def get_full_embeddings(cluster_model, descriptor_model, frame_data,
                        min_cluster_line_count=4,
                        max_cluster_line_count=50):
    scenes = frame_data.scenes
    floor_ids = get_floor_ids(scenes)
    training_plan = frame_data.training_plan

    # Compute the SIFT vocubulary used in the bag-of-words approach.
    vocabulary_dim = SIFT_VOCABULARY_DIM
    kmeans = get_sift_vocabulary(DATASET_PATH, SIFT_VOCABULARY_PATH, frame_data, vocabulary_dim)
    sift = cv2.xfeatures2d.SIFT_create()

    # In interiornet, different scenes can be in the same floor. These scenes are grouped into floors.
    query_floors = [QueryFloor([]) for i in range(np.unique(floor_ids).shape[0])]
    query_scenes = [QueryScene([]) for i in range(len(scenes))]

    total_nmi_nn = 0.
    total_nmi_agglo = 0.

    for idx, element in enumerate(training_plan):
        scene_id = element[0]
        floor_id = floor_ids[scene_id]
        frame_id = element[1]
        frame_path = scenes[scene_id].frame_paths[frame_id]

        # Get all geometries and virtual images from the frame.
        data, _ = frame_data.__getitem__(idx)
        frame_geometries = data['lines']
        frame_images = data['images']
        valid_mask = data['valid_input_mask']
        gt_labels = data['labels']

        # Get predicted embeddings using full prediction.
        # First, predict the cluster labels using the clustering network.
        cluster_output = cluster_model.predict_on_batch(data)
        cluster_output = cluster_output[0, valid_mask[0, :], :]
        cluster_output = np.argmax(cluster_output, axis=-1)

        if INFER_NMI:
            # If the NMI is to be computed, compute it for our method and for agglomerative clustering.
            nmi_nn = sm.normalized_mutual_info_score(gt_labels[0, valid_mask[0, :]], cluster_output)
            total_nmi_nn += nmi_nn
            _, nmi_agglo = inference_agglomerative.infer_on_frame(data)
            total_nmi_agglo += nmi_agglo
        else:
            nmi_nn = 0.
            nmi_agglo = 0.

        # Save the data into frames.
        query_frame = QueryFrame(frame_geometries[0, valid_mask[0, :], :], cluster_output, frame_path, [], [])

        # For each predicted cluster, compute the cluster descriptor.
        unique_output = np.unique(cluster_output)
        unique_output.sort()
        cluster_embeddings = []
        for label_idx in unique_output:
            line_indices = np.where(cluster_output == label_idx)[0][:max_cluster_line_count]
            cluster_line_count = len(line_indices)
            # Remove clusters with less lines than min_cluster_line_count.
            if cluster_line_count >= min_cluster_line_count:
                # Extract all lines of the corresponding label.
                cluster_lines = frame_geometries[0, line_indices[:cluster_line_count], :]
                # Add the mean back for rendering later.
                cluster_lines_render = unnormalize(cluster_lines.copy())
                # Set the mean of the lines of the cluster to zero for the neural network.
                cluster_lines = datagenerator_framewise.set_mean_zero(cluster_lines)

                # Preprocess the data for the neural network.
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
                # Predict the cluster descriptor embedding using the descriptor neural network.
                cluster_embedding = descriptor_model.predict_on_batch(cluster_data)
                cluster_embedding = np.array(cluster_embedding)[0, :]
                cluster_embeddings.append(cluster_embedding)

                # Save the cluster embedding and the cluster lines in the frame.
                query_frame.clusters.append(QueryCluster(cluster_lines_render, cluster_embedding, label_idx,
                                                         cluster_line_count))

        # Get embeddings from ground truth clustering.
        # Group lines by ground truth label.
        gt_labels = gt_labels[0, valid_mask[0, :]]
        unique_gt = np.unique(gt_labels)
        unique_gt.sort()
        gt_embeddings = []
        for j, label_idx in enumerate(unique_gt):
            line_indices = np.where(gt_labels == label_idx)[0][:max_cluster_line_count]
            cluster_line_count = len(line_indices)
            # Select only clusters that have more lines than the minimum line count.
            if cluster_line_count >= min_cluster_line_count:
                # Extract all lines of the corresponding label.
                cluster_lines = frame_geometries[0, line_indices[:cluster_line_count], :]
                # Add the mean back for rendering later.
                cluster_lines_render = unnormalize(cluster_lines.copy())
                # Set the mean of the lines of the cluster to zero for the neural network.
                cluster_lines = datagenerator_framewise.set_mean_zero(cluster_lines)

                # Preprocess the data for the neural network.
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
                # Predict the cluster descriptor embedding using the descriptor neural network.
                gt_embedding = descriptor_model.predict_on_batch(cluster_data)
                gt_embedding = np.array(gt_embedding)[0, :]
                gt_embeddings.append(gt_embedding)

                # Save the ground truth cluster embedding and the cluster lines in the frame.
                query_frame.gt_clusters.append(QueryCluster(cluster_lines_render, gt_embedding, j, cluster_line_count))

        # Get SIFT histogram for the frame.
        # Load the corresponding RGB image.
        scene_name = frame_path.split('/')[-2]
        frame_id = frame_path.split('_')[-1]
        image_path = os.path.join(DATASET_PATH, scene_name, 'cam0/data', frame_id + '.png')
        frame_im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if frame_im is None:
            print(image_path)

        # Compute the SIFT histogram.
        keypoints, descriptors = sift.detectAndCompute(frame_im, None)
        if descriptors is not None:
            indices = kmeans.predict(descriptors)
            sift_embedding = np.histogram(indices, np.arange(vocabulary_dim))[0].astype(float)
            # Add the SIFT embedding to the frame.
            query_frame.sift_embedding = sift_embedding
        else:
            query_frame.sift_embedding = None

        # Add the frame to the scene and the scene to the floor.
        query_scenes[scene_id].frames.append(query_frame)
        if query_scenes[scene_id] not in query_floors[floor_id].scenes:
            query_floors[floor_id].scenes.append(query_scenes[scene_id])

        print("Scene {}, frame {}, number of clusters: {}, gt clusters: {}, nmi nn: {}, nmi agglo: {}".format(scene_id,
                                                                                   frame_id,
                                                                                   len(cluster_embeddings),
                                                                                   len(gt_embeddings),
                                                                                   nmi_nn, nmi_agglo))

    # Save the computed NMIs for the neural network and agglomerative clustering in the file.
    if not os.path.exists('results'):
        os.mkdir('results')
    with open("results/nmis.txt", 'w') as f:
        f.write("NMI of neural network: {} \n".format(total_nmi_nn / len(training_plan)))
        f.write("NMI of agglomerative clustering: {} \n".format(total_nmi_agglo / len(training_plan)))

    return query_floors


def query_on_floors(query_floors):
    # Get fully predicted, gt clustered and sift embeddings.

    # The maximum number of floors to be used during the place recognition experiments.
    max_num_floors = MAX_NUM_FLOORS

    # Minimum and maximum frames per line
    min_frames = MIN_FRAMES_PER_SCENE
    max_frames = MAX_FRAMES_PER_SCENE

    # Minimum number of lines for the clusters used in place recognition.
    description_min_line_count = MIN_CLUSTER_LINES_FOR_DESCRIPTION
    # Number of nearest neighbors used during place recognition.
    k = K_NN

    sift_data = []
    gt_data = []
    data = []

    embeddings = []
    gt_embeddings = []
    sift_embeddings = []

    # Remove clusters with too few lines, remove scenes with too few frames and remove frames
    # so that the number of frames is less than max_frames.
    for floor in query_floors:
        floor.scenes[:] = [scene for scene in floor.scenes if len(scene.frames) >= min_frames]
        for scene in floor.scenes:
            if len(scene.frames) > max_frames:
                scene.frames[:] = np.random.choice(scene.frames, max_frames, replace=False)
            for frame in scene.frames:
                frame.clusters[:] = [cluster for cluster in frame.clusters
                                     if cluster.line_count >= description_min_line_count]
                frame.gt_clusters[:] = [cluster for cluster in frame.gt_clusters
                                        if cluster.line_count >= description_min_line_count]
    # Remove scenes so that the number of scenes is less than max_num_floors.
    if len(query_floors) > max_num_floors:
        query_floors[:] = np.random.choice(query_floors, max_num_floors, replace=False)

    # The scenes that should be removed because they do not appear in InteriorNet training.
    # This is mostly for the NYU dataset.
    blacklist = ["classroom", "computer_lab", "bathroom", "bookstore", "cafe", "excercise_room",
                 "office", "study_room", "basement", "office_kitchen"]
    whitelist = ["home_office"]

    # List all floors and scenes so that they can be queried later.
    number_of_scenes = 0
    for floor in query_floors[:max_num_floors]:
        for scene in floor.scenes:
            # Remove scenes that do not appear in InteriorNet training.
            wrong_scene = False
            for blacklisted in blacklist:
                if blacklisted in scene.frames[0].path:
                    wrong_scene = True
            for whitelisted in whitelist:
                if whitelisted in scene.frames[0].path:
                    wrong_scene = False
            if wrong_scene:
                continue

            for frame in scene.frames:
                for cluster in frame.clusters:
                    embeddings.append(cluster.embedding)
                    data.append((floor, scene, frame, cluster))
                for gt_cluster in frame.gt_clusters:
                    gt_embeddings.append(gt_cluster.embedding)
                    gt_data.append((floor, scene, frame, gt_cluster))
                sift_embeddings.append(frame.sift_embedding)
                sift_data.append((floor, scene, frame))
            number_of_scenes += 1

    print("Number of scenes: {}".format(number_of_scenes))
    print("k = {}".format(k))

    # Generate map trees for faster query.
    embeddings = np.vstack(embeddings)
    gt_embeddings = np.vstack(gt_embeddings)
    sift_embeddings = np.vstack(sift_embeddings)
    print("Generating map trees.")
    map_tree = sn.KDTree(embeddings, leaf_size=10)
    gt_tree = sn.KDTree(gt_embeddings, leaf_size=10)
    sift_tree = sn.KDTree(sift_embeddings, leaf_size=10)

    rendered_pred_matches = 0
    rendered_clusterings = 0
    max_render = 100

    # Create directories where the rendered images are to be saved.
    if RENDER:
        if not os.path.exists('visualization'):
            os.mkdir('visualization')
        if not os.path.exists('visualization/matches'):
            os.mkdir('visualization/matches')
        if not os.path.exists('visualization/matches_top10'):
            os.mkdir('visualization/matches_top10')
        if not os.path.exists('visualization/pred_clusters'):
            os.mkdir('visualization/pred_clusters')
        if not os.path.exists('visualization/gt_clusters'):
            os.mkdir('visualization/gt_clusters')
        if not os.path.exists('visualization/gt_matches'):
            os.mkdir('visualization/gt_matches')

    print("Starting query.")

    matched_num_clusters = 0
    tot_num_clusters = 0
    matched_num_frames = 0
    empty_num_frames = 0
    tot_num_frames = 0

    tot_num_gt_clusters = 0
    matched_num_gt_clusters = 0
    matched_num_gt_frames = 0
    matched_num_only_cluster = 0

    num_sift_frames = 0
    matched_num_sift_frames = 0
    for floor in query_floors[:max_num_floors]:
        for scene in floor.scenes:
            for frame in scene.frames:
                # Skip unwanted scenes.
                wrong_scene = False
                for blacklisted in blacklist:
                    if blacklisted in frame.path:
                        wrong_scene = True
                for whitelisted in whitelist:
                    if whitelisted in frame.path:
                        wrong_scene = False

                if wrong_scene:
                    continue

                # Perform place recognition using fully predicted clusters.
                frame_matches = []
                for i, cluster in enumerate(frame.clusters):
                    # Query the KD tree for the nearest neighbor.
                    dist, nearest = map_tree.query(cluster.embedding.reshape(-1, 1).T, k=k*2+1)
                    # Warning, the nearest one is of course the very same cluster and is removed.
                    nearest_data = [data[nearest[0][j]] for j in range(1, k*2+1)]
                    matched = False
                    cluster_matches = []
                    num_queries = 0
                    # Find all k nearest neighbors.
                    for data_point in nearest_data:
                        # Repeat until the number of nearest neighbors is k.
                        if num_queries >= k:
                            break
                        if data_point[2] is frame:
                            # If the matching cluster is inside the frame, ignore that match.
                            ...
                        else:
                            if data_point[0] is floor:
                                matched = True
                            # Add all nearest neighbors that are not inside the frame to the cluster_matches.
                            cluster_matches.append(query_floors.index(data_point[0]))
                            num_queries += 1
                    if matched:
                        matched_num_clusters += 1

                    frame_matches += cluster_matches
                    tot_num_clusters += 1

                    # Render the cluster matches if the maximum number of rendered clusters is not reached.
                    if rendered_clusterings < max_render and RENDER:
                        image_1 = frame.draw_frame(i)
                        image_2 = nearest_data[0][2].draw_frame(nearest_data[0][2].clusters.index(nearest_data[0][3]))
                        match_img = np.concatenate([image_1, image_2], axis=1)
                        cv2.imwrite("visualization/matches/{}_{}.png".format(rendered_clusterings,
                                                                             rendered_pred_matches),
                                    match_img,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

                        # Render all top k matches in one image. This might take a lot of time.
                        for j in range(1, k):
                            next_img = nearest_data[j][2].draw_frame(
                                nearest_data[j][2].clusters.index(nearest_data[j][3]))
                            match_img = np.concatenate([match_img, next_img], axis=1)
                        cv2.imwrite("visualization/matches_top10/{}_{}_{}.png".format(rendered_clusterings,
                                                                                      rendered_pred_matches,
                                                                                      matched),
                                    match_img,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 5])

                        rendered_pred_matches += 1

                # Check for the scene by majority voting.
                # If the scene is the correct scene, add this to the counter.
                cluster_matched = False
                if len(frame_matches) > 0:
                    most_common_matches = Counter(frame_matches).most_common(1)
                    if query_floors[most_common_matches[0][0]] is floor:
                        matched_num_frames += 1
                        cluster_matched = True
                    else:
                        ...
                else:
                    print("Empty frame.")
                    empty_num_frames += 1

                # Render frame with the ground truth and predicted clusterings.
                if rendered_clusterings < max_render and RENDER:
                    pred_img = frame.draw_frame()
                    gt_img = frame.draw_frame(predicted=False)
                    cv2.imwrite("visualization/pred_clusters/{}_{}.png".format(rendered_clusterings, cluster_matched),
                                pred_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imwrite("visualization/gt_clusters/{}_{}.png".format(rendered_clusterings,
                                                                             frame.path.split('/')[-1]), gt_img,
                                [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    rendered_clusterings += 1

                # Perform place recognition using ground truth clusterings. This works exactly the same way as above
                # with the only difference of using ground truth clusters instead of predicted clusters.
                gt_frame_matches = []
                for i, gt_cluster in enumerate(frame.gt_clusters):
                    dist, nearest = gt_tree.query(gt_cluster.embedding.reshape(-1, 1).T, k=k*2+1)
                    # Warning, the nearest one is of course the very same cluster.
                    nearest_data = [gt_data[nearest[0][j]] for j in range(1, k*2+1)]
                    matched = False
                    cluster_matches = []
                    num_queries = 0
                    for data_point in nearest_data:
                        if num_queries >= k:
                            break
                        if data_point[2] is frame:
                            ...
                        else:
                            if data_point[0] is floor:
                                matched = True
                            cluster_matches.append(query_floors.index(data_point[0]))
                            num_queries += 1
                    if matched:
                        matched_num_gt_clusters += 1

                    gt_frame_matches += cluster_matches
                    tot_num_gt_clusters += 1

                    # Render ground truth matches.
                    if rendered_clusterings < max_render and RENDER:
                        image_1 = frame.draw_frame(i, predicted=False)
                        image_2 = nearest_data[0][2].draw_frame(
                            nearest_data[0][2].gt_clusters.index(nearest_data[0][3]),
                            predicted=False)
                        match_img = np.concatenate([image_1, image_2], axis=1)
                        cv2.imwrite("visualization/gt_matches/{}_{}.png".format(rendered_clusterings,
                                                                                rendered_pred_matches),
                                    match_img)

                # Majority voting for predicted scene.
                if len(gt_frame_matches) > 0:
                    most_common_matches = Counter(gt_frame_matches).most_common(1)
                    if query_floors[most_common_matches[0][0]] is floor:
                        matched_num_gt_frames += 1

                # Perform place recognition with the SIFT bag-of-words approach.
                if frame.sift_embedding is not None:
                    k_sift = 1
                    dist, ind = sift_tree.query(frame.sift_embedding.reshape(-1, 1).T, k=k_sift+1)
                    # The nearest neighbor is chosen as the predicted scene.
                    nearest_data = [sift_data[ind[0][j]] for j in range(1, k_sift+1)]
                    sift_matches = []
                    for data_point in nearest_data:
                        sift_matches.append(query_floors.index(data_point[0]))
                    most_common_sift = Counter(sift_matches).most_common(1)
                    if query_floors[most_common_sift[0][0]] is floor:
                        matched_num_sift_frames += 1
                    elif cluster_matched:
                        matched_num_only_cluster += 1
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
    print("Number of frames not detected by SIFT but by line clustering: {}".format(matched_num_only_cluster))
    print("==================================================")


if __name__ == '__main__':
    # Set the random seed for reproducibility.
    np.random.seed(123)

    # The name of the dataset used.
    dataset_name = "interiornet"

    # The path to the saved weights of the cluster description model.
    embedding_model_path = \
        "/clustering_and_description/logs/description_100620_1644/weights_only.27.hdf5"
    # The path to the saved weights of the clustering model.
    cluster_model_path = \
        "/clustering_and_description/logs/cluster_110620_2345/weights_only.26.hdf5"
    # The path to the pickled precomputed map (sift histograms and cluster descriptors) for all frames in the map.
    PICKLE_PATH = "/nvme/line_ws/val_map"
    # The path to the preprocessed dataset directory containing the line files for the frames of the scenes used for
    # the place recognition experiments.
    MAP_DIR = "/nvme/line_ws/val_map"
    # The path to the (original) dataset directory in the interiornet format.
    DATASET_PATH = "/nvme/datasets/interiornet"
    # The path to the pickle file where the sift vocabulary should be saved to and loaded from.
    SIFT_VOCABULARY_PATH = "/nvme/line_ws/val_map"

    # The maximum number of floors to use during the experiment.
    MAX_NUM_FLOORS = 10000
    # The minimum number of frames per scene. Scenes with less frames are ignored.
    MIN_FRAMES_PER_SCENE = 0
    # The maximum number of frames per scene. Frames are removed from scenes to enforce this number.
    MAX_FRAMES_PER_SCENE = 100
    # The dimension of SIFT vocabulary used for the bag-of-words approach.
    SIFT_VOCABULARY_DIM = 800
    # The number of nearest neighbors to be considered for our approach.
    K_NN = 8
    # The minimum number of lines for each cluster to be used in our approach.
    MIN_CLUSTER_LINES_FOR_DESCRIPTION = 4
    # The background class ids used for ground truth clustering. All background lines are put into one cluster.
    BG_CLASSES = [0, 1, 2, 20, 22]

    # INTERIORNET:
    if dataset_name == "interiornet":
        PICKLE_PATH = "/nvme/line_ws/val_map"
        MAP_DIR = "/nvme/line_ws/val_map"
        DATASET_PATH = "/nvme/datasets/interiornet"
        SIFT_VOCABULARY_PATH = "/nvme/line_ws/val_map/sift_vocabulary"

        MAX_NUM_FLOORS = 10000
        MIN_FRAMES_PER_SCENE = 0
        MAX_FRAMES_PER_SCENE = 100
        SIFT_VOCABULARY_DIM = 800
        K_NN = 8
        MIN_CLUSTER_LINES_FOR_DESCRIPTION = 4
        BG_CLASSES = [0, 1, 2, 20, 22]

    # DIML:
    elif dataset_name == "diml":
        PICKLE_PATH = "/nvme/line_ws/all_data_diml"
        MAP_DIR = "/nvme/line_ws/all_data_diml"
        DATASET_PATH = "/nvme/datasets/diml_depth/HD7"
        SIFT_VOCABULARY_PATH = "/nvme/line_ws/all_data_diml/sift_vocabulary"

        MAX_NUM_FLOORS = 10000
        MIN_FRAMES_PER_SCENE = 8
        MAX_FRAMES_PER_SCENE = 8
        SIFT_VOCABULARY_DIM = 512
        K_NN = 8
        MIN_CLUSTER_LINES_FOR_DESCRIPTION = 5
        BG_CLASSES = []

    # NYU:
    elif dataset_name == "nyu":
        PICKLE_PATH = "/nvme/line_nyu/all_data_4_or_more"
        MAP_DIR = PICKLE_PATH
        DATASET_PATH = "/nvme/datasets/nyu_v2/HD7"
        SIFT_VOCABULARY_PATH = "/nvme/line_nyu/all_data_4_or_more/sift_vocabulary"

        MAX_NUM_FLOORS = 10000
        MIN_FRAMES_PER_SCENE = 5
        MAX_FRAMES_PER_SCENE = 8
        SIFT_VOCABULARY_DIM = 512
        K_NN = 4
        MIN_CLUSTER_LINES_FOR_DESCRIPTION = 5
        BG_CLASSES = []

    else:
        print("WARNING: No dataset specified, using default values.")

    # RENDER indicates if the results of the descriptor inference should be rendered as images.
    # Warning, rendering is very slow.
    RENDER = False
    # INFER_NMI indicates if the NMI of the clustering methods should be computed or not.
    INFER_NMI = False

    line_num_attr = 15
    img_shape = (64, 96, 3)
    # The number of lines for the neural networks during precomputation.
    # The actual minimum number of lines for cluster description during usage in place recognition
    # is determined by MIN_CLUSTER_LINES_FOR_DESCRIPTION
    max_cluster_line_count = 70
    min_cluster_line_count = 2
    max_frame_line_count = 220
    min_frame_line_count = 20
    # Other hyper-parameters for the neural networks.
    max_clusters = 15
    batch_size = 1
    margin = 0.3
    embedding_dim = 128
    # All lines from background classes will be combined into one line cluster during ground truth clustering.
    valid_classes = [i for i in range(41) if i not in BG_CLASSES]

    # The path where the precomputed map will be saved.
    pickle_name = "map_pickle"
    PICKLE_PATH = os.path.join(PICKLE_PATH, pickle_name)

    # Check if precomputed map already exists.
    if os.path.isfile(PICKLE_PATH):
        print("Found map, loading.")
        with open(PICKLE_PATH, 'rb') as f:
            # If the precomputed map already exists, use that one.
            query_floors = pickle.load(f)
    else:
        print("Loading data.")
        print("Using end to end clustering.")

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Load the neural networks and weights.
        line_model = model.load_line_net_model(cluster_model_path, line_num_attr, max_frame_line_count,
                                               max_clusters, img_shape)
        embedding_model = model.load_cluster_embedding_model(embedding_model_path, line_num_attr, max_cluster_line_count,
                                                             embedding_dim, img_shape, margin)

        # Initialize the framewise datagenerator for the dataset.
        frame_data = datagenerator_framewise.LineDataSequence(MAP_DIR,
                                                              batch_size,
                                                              BG_CLASSES,
                                                              data_augmentation=False,
                                                              img_shape=img_shape,
                                                              min_line_count=min_frame_line_count,
                                                              max_line_count=max_frame_line_count,
                                                              max_cluster_count=max_clusters,
                                                              training_mode=False)

        # Start computing the map, by computing all SIFT bag of words and all cluster descriptors.
        query_floors = get_full_embeddings(line_model, embedding_model, frame_data,
                                           min_cluster_line_count=min_cluster_line_count,
                                           max_cluster_line_count=max_cluster_line_count)

        # Save map for later use.
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(query_floors, f)

    print("Starting query.")

    # Set seed for reproducibility during query.
    np.random.seed(128)

    # Start place recognition experiments.
    query_on_floors(query_floors)



