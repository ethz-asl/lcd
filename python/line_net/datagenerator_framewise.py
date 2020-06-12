import numpy as np
import pandas as pd
import cv2
import os
import sys
import time
import pickle
from numba import jit, njit
from collections import Counter

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import Sequence


def load_frame(path):
    data_lines = pd.read_csv(path, sep=" ", header=None)
    line_vci_paths = data_lines.values[:, 0]
    line_geometries = data_lines.values[:, 1:15].astype(float)
    line_labels = data_lines.values[:, 15].astype(int)
    line_class_ids = data_lines.values[:, 17].astype(int)
    line_count = line_geometries.shape[0]

    cam_origin = data_lines.values[0, 19:22].astype(float)
    cam_rotation = data_lines.values[0, 22:26].astype(float)
    transform = get_world_transform(cam_origin, cam_rotation)

    return line_count, line_geometries, line_labels, line_class_ids, line_vci_paths, transform


def get_world_transform(cam_origin, cam_rotation):
    x = cam_origin[0]
    y = cam_origin[1]
    z = cam_origin[2]
    e0 = cam_rotation[3]
    e1 = cam_rotation[0]
    e2 = cam_rotation[1]
    e3 = cam_rotation[2]
    T = np.array([[e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * e1 * e2 - 2 * e0 * e3, 2 * e0 * e2 + 2 * e1 * e3, x],
                  [2 * e0 * e3 + 2 * e1 * e2, e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * e2 * e3 - 2 * e0 * e1, y],
                  [2 * e1 * e3 - 2 * e0 * e2, 2 * e0 * e1 + 2 * e2 * e3, e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2, z],
                  [0, 0, 0, 1]])
    return T


def set_mean_zero(lines):
    mean = np.mean(np.vstack([lines[:, :3], lines[:, 3:6]]), axis=0)
    lines[:, :3] -= mean
    lines[:, 3:6] -= mean

    return lines


def transform_lines(lines, transform):
    lines[:, :3] = transform.dot(np.vstack([lines[:, :3].T, np.ones((1, lines.shape[0]))]))[:3, :].T
    lines[:, 3:6] = transform.dot(np.vstack([lines[:, 3:6].T, np.ones((1, lines.shape[0]))]))[:3, :].T
    lines[:, 6:9] = transform[:3, :3].dot(lines[:, 6:9].T).T
    lines[:, 9:12] = transform[:3, :3].dot(lines[:, 9:12].T).T

    return lines


def fuse_frames(geom_1, labels_1, class_1, vcis_1, geom_2, labels_2, class_2, vcis_2):
    groups = group_lines(geom_1, geom_2)

    out_geometries = []
    out_labels = []
    out_classes = []
    out_vcis = []

    for group in groups:
        geometries = [geom_1[i, :] for i in group[0]] + [geom_2[j, :] for j in group[1]]
        labels = [labels_1[i] for i in group[0]] + [labels_2[j] for j in group[1]]
        classes = [class_1[i] for i in group[0]] + [class_2[j] for j in group[1]]
        vcis = [vcis_1[i] for i in group[0]] + [vcis_2[j] for j in group[1]]

        # Fuse lines by geometry
        fused_line = fuse_line_group(geometries)

        fused_label = Counter(labels).most_common(1)[0][0]
        fused_class = Counter(classes).most_common(1)[0][0]

        resolutions = np.array([vci.shape[0] for vci in vcis])
        fused_vci = vcis[int(np.argmax(resolutions))]

        out_geometries.append(fused_line)
        out_labels.append(fused_label)
        out_classes.append(fused_class)
        out_vcis.append(fused_vci)

    return out_geometries, out_labels, out_classes, out_vcis


def group_lines(lines_1, lines_2):
    groups = [[{j}, set()] for j in range(lines_1.shape[0])] + \
        [[set(), {i}] for i in range(lines_2.shape[0])]

    for i in range(lines_2.shape[0]):
        line = lines_2[i, :]
        for j in range(lines_1.shape[0]):
            if lines_coincide(lines_1[j, :], line):
                grouped = []
                for k, pair in enumerate(groups):
                    if i in pair[1] or j in pair[0]:
                        grouped.append(k)

                new_group = [{j}, {i}]
                for g in grouped:
                    new_group[0] = new_group[0] | groups[g][0]
                    new_group[1] = new_group[1] | groups[g][1]

                for g in sorted(grouped, reverse=True):
                    del(groups[g])
                groups.append(new_group)

    return groups


@njit
def lines_coincide(line_1, line_2):
    # tic = time.perf_counter()
    max_angle = 0.15
    max_dis = 0.015

    start_1 = line_1[0:3]
    end_1 = line_1[3:6]
    dir_1 = end_1 - start_1
    l_1 = np.linalg.norm(dir_1)
    dir_1_n = dir_1 / l_1

    start_2 = line_2[0:3]
    end_2 = line_2[3:6]
    dir_2 = end_2 - start_2
    l_2 = np.linalg.norm(dir_2)
    dir_2_n = dir_2 / l_2

    # Check if the angle of the line is not above a certain threshold.
    angle = np.abs(np.dot(dir_1_n, dir_2_n))

    # print("Calculating took {} seconds.".format(time.perf_counter() - tic))
    if angle > np.cos(max_angle):
        # Check if the orthogonal distance between the lines are lower than a certain threshold.
        dis_3 = np.linalg.norm(np.cross(dir_1_n, start_2 - start_1))
        dis_4 = np.linalg.norm(np.cross(dir_1_n, end_2 - start_1))

        if dis_3 < max_dis or dis_4 < max_dis:
            # Check if the lines overlap.
            x_3 = np.dot(dir_1_n, start_2 - start_1)
            x_4 = np.dot(dir_1_n, end_2 - start_1)
            if min(x_3, x_4) < 0. < max(x_3, x_4) or 0. < min(x_3, x_4) < l_1:
                return True

    return False


def fuse_line_group(lines):
    start_1 = lines[0][:3]
    end_1 = lines[0][3:6]
    l_1 = np.linalg.norm(end_1 - start_1)
    dir_1 = (end_1 - start_1) / l_1
    start_1_open = lines[0][12]
    end_1_open = lines[0][13]

    x = [0., l_1]
    points = [start_1, end_1]
    opens = [start_1_open, end_1_open]

    for line in lines[1:]:
        x.append(dir_1.dot(line[:3] - start_1))
        points.append(line[:3])
        opens.append(line[12])
        x.append(dir_1.dot(line[3:6] - start_1))
        points.append(line[3:6])
        opens.append(line[13])

    start_idx = int(np.argmin(x))
    end_idx = int(np.argmax(x))
    new_start = points[start_idx]
    new_end = points[end_idx]
    new_start_open = opens[start_idx]
    new_end_open = opens[end_idx]

    new_normal_1 = lines[0][6:9]
    new_normal_2 = lines[0][9:12]
    # Find a line that has two normals, and use those.
    for line in lines:
        n_1 = line[6:9]
        n_2 = line[9:12]
        if not (n_1 == 0.).all() and not (n_2 == 0.).all():
            new_normal_1 = n_1
            new_normal_2 = n_2
            break

    return np.hstack([new_start, new_end, new_normal_1, new_normal_2, new_start_open, new_end_open])


# Not used.
def fuse_lines(line_1, line_2):
    max_angle = 0.05
    max_dis = 0.025
    max_normal_angle = 0.1

    start_1 = line_1[:3]
    end_1 = line_1[3:6]
    dir_1 = end_1 - start_1
    l_1 = np.linalg.norm(dir_1)
    dir_1_n = dir_1 / l_1

    start_2 = line_2[:3]
    end_2 = line_2[3:6]
    dir_2 = end_2 - start_2
    l_2 = np.linalg.norm(dir_2)
    dir_2_n = dir_2 / l_2

    # Check if the angle of the line is not above a certain threshold.
    angle = np.abs(np.dot(dir_1_n, dir_2_n))
    if angle > np.cos(max_angle):
        # Check if the orthogonal distance between the lines are lower than a certain threshold.
        dis_3 = np.linalg.norm(np.cross(dir_1, start_2 - start_1))
        dis_4 = np.linalg.norm(np.cross(dir_1, end_2 - start_1))

        if dis_3 < max_dis and dis_4 < max_dis:
            # Check if the lines overlap.
            x_3 = np.dot(dir_1, start_2 - start_1)
            x_4 = np.dot(dir_1, end_2 - start_1)
            if min(x_3, x_4) < 0 < max(x_3, x_4) or 0 < min(x_3, x_4) < l_1:
                # We have an overlapping line!
                new_start_p = start_1
                new_start_open = line_1[-2]
                new_end_p = end_1
                new_end_open = line_1[-1]
                if x_3 < x_4:
                    if x_3 < 0:
                        new_start_p = start_2
                    elif x_4 > l_1:
                        new_end_p = end_2
                elif x_4 < x_3:
                    if x_4 < 0:
                        new_start_p = end_2
                    elif x_3 > l_1:
                        new_end_p = start_2

                # Find common normals.
                normal_1_1 = line_1[6:9]
                normal_1_2 = line_1[9:12]
                normal_2_1 = line_2[6:9]
                normal_2_2 = line_2[9:12]
                angle_1_1 = normal_1_1.dot(normal_2_1)
                angle_1_2 = normal_1_1.dot(normal_2_2)
                angle_2_1 = normal_1_2.dot(normal_2_1)
                angle_2_2 = normal_1_2.dot(normal_2_2)

                if angle_1_1 > angle_2_1 > np.cos(max_normal_angle):
                    # Merge 1 and 1, 2 and 2
                    new_normal_1 = fuse_normals(normal_1_1, normal_2_1, l_1, l_2, angle_1_1, max_normal_angle)
                    new_normal_2 = fuse_normals(normal_1_2, normal_2_2, l_1, l_2, angle_2_2, max_normal_angle)
                elif angle_2_1 > angle_1_1 > np.cos(max_normal_angle):
                    # Merge 2 and 1, 1 and 2
                    new_normal_1 = fuse_normals(normal_1_2, normal_2_1, l_1, l_2, angle_2_1, max_normal_angle)
                    new_normal_2 = fuse_normals(normal_1_1, normal_2_2, l_1, l_2, angle_1_2, max_normal_angle)
                elif angle_1_2 > angle_2_2 > np.cos(max_normal_angle):
                    # Merge 1 and 2, 2 and 1
                    new_normal_1 = fuse_normals(normal_1_1, normal_2_2, l_1, l_2, angle_1_2, max_normal_angle)
                    new_normal_2 = fuse_normals(normal_1_2, normal_2_1, l_1, l_2, angle_2_1, max_normal_angle)
                elif angle_2_2 > angle_1_2 > np.cos(max_normal_angle):
                    # Merge 2 and 2, 1 and 1
                    new_normal_1 = fuse_normals(normal_1_2, normal_2_2, l_1, l_2, angle_2_2, max_normal_angle)
                    new_normal_2 = fuse_normals(normal_1_1, normal_2_1, l_1, l_2, angle_1_1, max_normal_angle)
                else:
                    if l_1 > l_2:
                        new_normal_1 = normal_1_1
                        new_normal_2 = normal_1_2
                    else:
                        new_normal_1 = normal_2_1
                        new_normal_2 = normal_2_2

                # Return the fused line.
                return np.hstack([new_start_p, new_end_p, new_normal_1, new_normal_2, new_start_open, new_end_open])

    # If the lines do not match, return None.
    return None


def fuse_normals(normal_1, normal_2, length_1, length_2, angle_1_2, max_angle):
    # If one normal does not exist, return the other one.
    if (normal_1 == 0.).all():
        return normal_2
    if (normal_2 == 0.).all():
        return normal_1

    if angle_1_2 > np.cos(max_angle):
        # Return the interpolated normal.
        normal = normal_1 + normal_2
        return normal / np.linalg.norm(normal)
    else:
        # Return the normal of the longest line.
        if length_1 > length_2:
            return normal_1
        else:
            return normal_2


def load_image(path, img_shape):
    # FAIL: This reads images as BGR, but we want RGB for the pretrained vgg net...
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("WARNING: VIRTUAL CAMERA IMAGE NOT FOUND AT {}".format(path))
        return np.zeros(img_shape)
    else:
        img = preprocess_input(img)
        img = cv2.resize(img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img


class Cluster:
    def __init__(self, frame_id, frame_path_id, lines):
        self.lines = lines
        self.frame_id = frame_id
        self.frame_path_id = frame_path_id


class Scene:
    def __init__(self, path, bg_classes, min_line_count, max_clusters, img_shape, valid_classes,
                 min_line_count_cluster=5):
        self.scene_path = path
        self.name = path.split('/')[-1]
        self.frame_paths = [os.path.join(path, name) for name in os.listdir(path)
                            if os.path.isfile(os.path.join(path, name))]
        self.frame_paths.sort(key=lambda x: int(x.split('_')[-1]))
        self.frame_count = len(self.frame_paths)
        self.bg_classes = bg_classes
        self.valid_classes = valid_classes
        self.min_line_count = min_line_count
        self.max_clusters = max_clusters
        self.img_shape = img_shape
        self.valid_frames = []
        self.clusters = []
        self.cluster_labels = []
        self.cluster_classes = []
        self.min_line_count_cluster = min_line_count_cluster
        self.get_frames_info()

    def get_frames_info(self):
        cluster_labels = []
        cluster_classes = []
        clusters = []

        for i, frame in enumerate(self.frame_paths):
            frame_idx = int(frame.split('_')[-1])
            count, geometries, labels, classes, vci_paths, transform = load_frame(frame)
            frame_clusters = Counter(labels[get_non_bg_mask(classes, self.bg_classes)]).most_common()

            if count > self.min_line_count and len(frame_clusters) > 0:
                self.valid_frames.append(i)

            for cluster in frame_clusters:
                cluster_line_count = cluster[1]
                if cluster_line_count >= self.min_line_count_cluster:
                    cluster_label = cluster[0]
                    cluster_class = classes[np.where(labels == cluster_label)][0]
                    cluster_lines = np.where(labels == cluster_label)

                    if cluster_class in self.bg_classes:
                        cluster_label = 0
                    elif cluster_class not in self.valid_classes:
                        continue

                    new_cluster = Cluster(i, frame_idx, cluster_lines)
                    if cluster_label in cluster_labels:
                        clusters[cluster_labels.index(cluster_label)].append(new_cluster)
                    else:
                        cluster_labels.append(cluster_label)
                        cluster_classes.append(cluster_class)
                        clusters.append([new_cluster])

        self.clusters = clusters
        self.cluster_labels = cluster_labels
        self.cluster_classes = cluster_classes

    def get_cluster(self, cluster_id, line_count, shuffle, blacklisted=-1, center=True, forced_choice=None):
        if forced_choice is None:
            cluster_count_at_id = len(self.clusters[cluster_id])
            if shuffle:
                cluster_choice = np.random.choice([x for x in range(cluster_count_at_id) if x != blacklisted])
            else:
                cluster_choice = blacklisted + 1
        else:
            cluster_choice = forced_choice

        cluster = self.clusters[cluster_id][cluster_choice]
        cluster_label = self.cluster_labels[cluster_id]
        cluster_class = self.cluster_classes[cluster_id]
        frame_id = cluster.frame_id

        tot_count, lines, labels, classes, vci_paths, t_1 = load_frame(self.frame_paths[frame_id])

        if cluster_class in self.bg_classes:
            # If we have a background label, all background lines are put into the cluster.
            indices = np.where(np.isin(classes, self.bg_classes))[0]
        else:
            indices = np.where(labels == cluster_label)[0]

        if shuffle:
            np.random.shuffle(indices)

        lines = lines[indices[:line_count], :]
        images = np.concatenate([np.expand_dims(load_image(vci_paths[i], self.img_shape), 0)
                                 for i in indices[:line_count]])
        labels = labels[indices[:line_count]]
        classes = classes[indices[:line_count]]

        if cluster_class not in self.bg_classes:
            if np.unique(classes).shape[0] != 1:
                print("WARNING: MORE THAN 1 CLASS DETECTED.")
                print(classes)
                print(labels)
                print(self.frame_paths[frame_id])

        if center:
            lines = set_mean_zero(lines)

        return min(line_count, len(indices)), lines, cluster_label, cluster_class, images, cluster_choice

    def get_frame(self, frame_id, fusion_frames, line_count, shuffle):
        tot_count, lines, labels, classes, vci_paths, t_1 = load_frame(self.frame_paths[frame_id])
        vcis = [load_image(path, self.img_shape) for path in vci_paths]

        # tic = time.perf_counter()
        for fuse_frame_id in fusion_frames:
            count_2, lines_2, labels_2, classes_2, vci_paths_2, t_2 = load_frame(self.frame_paths[fuse_frame_id])
            vcis_2 = [load_image(path, self.img_shape) for path in vci_paths_2]
            lines_2 = transform_lines(lines_2, np.linalg.inv(t_1).dot(t_2))
            lines, labels, classes, vcis = fuse_frames(np.array(lines), np.array(labels), np.array(classes), vcis,
                                                       lines_2, labels_2, classes_2, vcis_2)
            tot_count += count_2
            vcis = vcis + vcis_2

        labels = np.stack(labels)
        lines = np.vstack(lines)
        classes = np.stack(classes)

        # Delete lines of small clusters so that the number of clusters is below maximum.
        cluster_line_counts = Counter(np.array(labels)[get_non_bg_mask(classes, self.bg_classes)]).most_common()
        for pair in cluster_line_counts[self.max_clusters:]:
            delete_idx = np.where(labels == pair[0])
            labels = np.delete(labels, delete_idx)
            lines = np.delete(lines, delete_idx, axis=0)
            classes = np.delete(classes, delete_idx)
            for i in np.sort(delete_idx[0])[::-1]:
                del(vcis[i])

        count = labels.shape[0]

        indices = np.arange(count)
        if shuffle:
            np.random.shuffle(indices)
        else:
            rand_state = np.random.get_state()
            np.random.seed(0)
            np.random.shuffle(indices)
            np.random.set_state(rand_state)
        indices = indices[:line_count]

        return count, lines[indices, :], labels[indices], classes[indices], \
            np.concatenate([np.expand_dims(img, axis=0) for img in np.array(vcis)[indices]], axis=0)


def get_non_bg_mask(classes, bg_classes):
    return np.where(np.isin(classes, bg_classes, invert=True))


class Frame:
    def __init__(self, path):
        self.path = path

    def get_batch(self, batch_size, img_shape, shuffle, load_images):
        self.line_count, self.line_geometries, self.line_labels, self.line_class_ids, self.line_vci_paths = \
            load_frame(self.path)

        count = min(batch_size, self.line_count)
        indices = np.arange(count)
        if shuffle:
            np.random.shuffle(indices)

        images = []

        if load_images:
            # Do not forget to shuffle images according to indices.
            for i in indices:
                img = cv2.imread(self.line_vci_paths[i], cv2.IMREAD_UNCHANGED)
                if img is None:
                    print("WARNING: VIRTUAL CAMERA IMAGE NOT FOUND AT {}".format(self.line_vci_paths[i]))
                    images.append(np.expand_dims(np.zeros(img_shape), axis=0))
                else:
                    img = preprocess_input(img)
                    img = cv2.resize(img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
                    images.append(np.expand_dims(img, axis=0))
            images = np.concatenate(images, axis=0)

        return count, self.line_geometries[indices, :], \
            self.line_labels[indices], self.line_class_ids[indices], images


def load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_clusters, img_shape, valid_classes,
                min_line_count_cluster=5):
    pickle_file_path = os.path.join(files_dir, "scenes_{}_{}_{}".format(min_line_count, max_line_count, max_clusters))
    if os.path.isfile(pickle_file_path):
        print("Opening existing scene file.")
        with open(pickle_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Creating new scene file.")
        scene_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isdir(os.path.join(files_dir, name))]
        scene_paths.sort()
        scenes = [Scene(path, bg_classes, min_line_count, max_clusters, img_shape, valid_classes,
                        min_line_count_cluster=min_line_count_cluster)
                  for path in scene_paths]
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(scenes, f)
        return scenes


def create_training_plan(scenes):
    plan = []
    for i, scene in enumerate(scenes):
        for frame_id in scene.valid_frames:
            plan.append((i, frame_id))

    return plan


class LineDataSequence(Sequence):
    def __init__(self, files_dir, batch_size, bg_classes, shuffle=False, fuse=False, data_augmentation=False,
                 mean=np.array([0., 0., 3.]), img_shape=(64, 96, 3), min_line_count=30, max_line_count=300,
                 max_cluster_count=30, load_images=True, training_mode=True):
        self.files_dir = files_dir
        self.bg_classes = bg_classes
        self.shuffle = shuffle
        self.fuse = fuse
        self.data_augmentation = data_augmentation
        self.mean = mean
        self.img_shape = img_shape
        self.load_images = load_images
        self.min_line_count = min_line_count
        self.max_line_count = max_line_count
        self.max_cluster_count = max_cluster_count
        self.batch_size = batch_size
        if training_mode:
            max_scene_cluster_count = max_cluster_count
        else:
            max_scene_cluster_count = 10000
        self.scenes = load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_scene_cluster_count,
                                  img_shape, [])
        self.training_plan = create_training_plan(self.scenes)
        self.frame_count = len(self.training_plan)
        self.frame_indices = np.arange(self.frame_count, dtype=int)
        self.shuffle_indices()

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.frame_indices)

    def __getitem__(self, idx):
        training_plan = [self.training_plan[i] for i in
                         self.frame_indices[idx * self.batch_size:(idx + 1) * self.batch_size]]

        batch_geometries = []
        batch_labels = []
        batch_valid_mask = []
        batch_bg_mask = []
        batch_images = []
        batch_unique = []
        batch_counts = []

        if not self.shuffle:
            rand_state = np.random.get_state()
            np.random.seed(idx)

        for element in training_plan:
            if self.fuse:
                num_fuses = np.random.randint(4)
                other_frames = np.arange(self.scenes[element[0]].frame_count)
                other_frames = np.delete(other_frames, element[1])
                other_frames = np.random.choice(other_frames, num_fuses, False)
            else:
                other_frames = []

            line_count, line_geometries, line_labels, line_class_ids, line_images = \
                self.scenes[element[0]].get_frame(element[1], other_frames, self.max_line_count, self.shuffle)

            line_geometries = subtract_mean(line_geometries, self.mean)

            if self.data_augmentation and np.random.binomial(1, 0.5):
                # augment_flip(line_geometries, line_images)
                augment_global(line_geometries, np.radians(15.), 0.3)
            if self.data_augmentation and np.random.binomial(1, 0.5):
                augment_images(line_images)

            line_geometries = normalize(line_geometries, 2.0)
            line_geometries = add_length(line_geometries)

            out_valid_mask = np.zeros((self.max_line_count,), dtype=bool)
            out_geometries = np.zeros((self.max_line_count, line_geometries.shape[1]))
            out_labels = np.zeros((self.max_line_count,), dtype=int)
            out_classes = np.zeros((self.max_line_count,), dtype=int)
            out_bg = np.zeros((self.max_line_count,), dtype=bool)

            out_valid_mask[:line_count] = True
            out_geometries[:line_count, :] = line_geometries
            out_classes[:line_count] = line_class_ids
            out_bg[line_count:self.max_line_count] = False
            out_bg[np.isin(out_classes, self.bg_classes)] = True
            for i, label in enumerate(np.unique(line_labels)):
                # We reserve label 0 for background lines.
                out_labels[np.where(line_labels == label)] = i + 1
            out_labels[out_bg] = 0

            if self.load_images:
                out_images = line_images
                if line_count < self.max_line_count:
                    out_images = np.concatenate([out_images,
                                                 np.zeros((self.max_line_count - line_count,
                                                           self.img_shape[0],
                                                           self.img_shape[1],
                                                           self.img_shape[2]))], axis=0)
            else:
                out_images = np.zeros((self.max_line_count, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

            batch_labels.append(np.expand_dims(out_labels, axis=0))
            batch_geometries.append(np.expand_dims(out_geometries, axis=0))
            batch_valid_mask.append(np.expand_dims(out_valid_mask, axis=0))
            batch_bg_mask.append(np.expand_dims(out_bg, axis=0))
            batch_images.append(np.expand_dims(out_images, axis=0))

            unique_labels = np.unique(out_labels[np.logical_and(np.logical_not(out_bg), out_valid_mask)])
            cluster_count = unique_labels.shape[0]
            if cluster_count >= self.max_cluster_count:
                unique_labels = unique_labels[:self.max_cluster_count]
            else:
                unique_labels = np.pad(unique_labels, (0, self.max_cluster_count - cluster_count),
                                       mode='constant', constant_values=-1)
            if cluster_count == 0:
                print("WARNING: NO CLUSTERS IN FRAME.")
            unique_labels = np.expand_dims(unique_labels, axis=0)
            cluster_count = np.expand_dims(cluster_count, axis=0)
            batch_unique.append(unique_labels)
            batch_counts.append(cluster_count)

        geometries = np.concatenate(batch_geometries, axis=0)
        labels = np.concatenate(batch_labels, axis=0)
        valid_mask = np.concatenate(batch_valid_mask, axis=0)
        bg_mask = np.concatenate(batch_bg_mask, axis=0)
        images = np.concatenate(batch_images, axis=0)
        unique = np.concatenate(batch_unique, axis=0)
        counts = np.concatenate(batch_counts, axis=0)

        if not self.shuffle:
            np.random.set_state(rand_state)

        return {'lines': geometries,
                'labels': labels,
                'valid_input_mask': valid_mask,
                'background_mask': bg_mask,
                'images': images,
                'unique_labels': unique,
                'cluster_count': counts}, labels

    def __len__(self):
        return int(np.ceil(self.frame_count / self.batch_size))


def create_training_plan_clusters(scenes, min_num_clusters=2):
    class_histogram = []

    plan = []

    # The floor names are the name of the floors with multiple scenes, e.g. 3FO4IMXSHSVT
    floor_names = []
    curr_id = 0
    floor_ids = np.zeros((len(scenes),), dtype=int)
    for i, scene in enumerate(scenes):
        if len(scene.cluster_labels) > 0:
            floor_name = scene.name.split('_')[0]
            if floor_name in floor_names:
                floor_ids[i] = floor_names.index(floor_name)
            else:
                floor_names.append(floor_name)
                floor_ids[i] = curr_id
                curr_id += 1

            # The white list is the the list of scenes that are not the scene the current cluster is in,
            # to prevent trying to distinct between instances that are the same.
            white_list = []
            for j, other_scene in enumerate(scenes):
                other_floor_name = other_scene.name.split('_')[0]
                if len(other_scene.cluster_labels) > 0:
                    if floor_name != other_floor_name:
                        white_list.append(j)
            for j, cluster_label in enumerate(scene.cluster_classes):
                if len(scene.clusters[j]) >= min_num_clusters:
                    plan.append((i, j, white_list, floor_ids[i]))
                    class_histogram.append(scene.cluster_classes[j])

    print("Found {} scenes in {} floors.".format(len(scenes), len(floor_names)))
    # import matplotlib.pyplot as plt
    # plt.hist(class_histogram, bins=np.arange(40))
    # plt.show()

    return plan, floor_ids


class ClusterDataSequence(Sequence):
    def __init__(self, files_dir, batch_size, bg_classes, classes, shuffle=False, data_augmentation=False,
                 img_shape=(64, 96, 3), min_line_count=5, max_line_count=50,
                 load_images=True, training_mode=True):
        self.files_dir = files_dir
        self.bg_classes = bg_classes
        self.classes = classes
        self.class_count = 1 + len(classes)
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.img_shape = img_shape
        self.load_images = load_images
        self.min_line_count = min_line_count
        self.max_line_count = max_line_count
        self.batch_size = batch_size
        self.scenes = load_scenes(files_dir, bg_classes, min_line_count, max_line_count, 1000, img_shape, classes,
                                  min_line_count_cluster=min_line_count)
        if training_mode:
            min_num_clusters = 2
        else:
            min_num_clusters = 1
        self.training_plan, self.floor_ids = create_training_plan_clusters(self.scenes, min_num_clusters)
        self.cluster_count = len(self.training_plan)
        self.cluster_indices = np.arange(self.cluster_count, dtype=int)
        self.shuffle_indices()

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.cluster_indices)

    def process_cluster(self, line_count, cluster_lines, cluster_label, cluster_class, cluster_images):
        if self.data_augmentation and np.random.binomial(1, 0.5):
            # augment_flip(cluster_lines, cluster_images)
            augment_global(cluster_lines, np.radians(8.), 0.0)
            augment_images(cluster_images)

        line_geometries = normalize(cluster_lines, 2.0)
        line_geometries = add_length(line_geometries)

        out_valid_mask = np.zeros((self.max_line_count,), dtype=bool)
        out_geometries = np.zeros((self.max_line_count, line_geometries.shape[1]))

        out_valid_mask[:line_count] = True
        out_geometries[:line_count, :] = line_geometries

        if self.load_images:
            out_images = cluster_images
            if line_count < self.max_line_count:
                out_images = np.concatenate([out_images,
                                             np.zeros((self.max_line_count - line_count,
                                                       self.img_shape[0],
                                                       self.img_shape[1],
                                                       self.img_shape[2]))], axis=0)
        else:
            out_images = np.zeros((self.max_line_count, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

        if cluster_class in self.bg_classes:
            cluster_class = 0
        else:
            cluster_class = self.classes.index(cluster_class) + 1

        # out_gt = np.zeros((self.class_count,))
        # out_gt[cluster_class] = 1.

        return np.expand_dims(out_geometries, axis=0), np.expand_dims(out_valid_mask, axis=0), \
               np.expand_dims(out_images, axis=0)

    def __getitem__(self, idx):
        training_plan = [self.training_plan[i] for i in
                         self.cluster_indices[idx * self.batch_size:(idx + 1) * self.batch_size]]

        batch_geometries_a = []
        batch_valid_mask_a = []
        batch_images_a = []
        batch_geometries_p = []
        batch_valid_mask_p = []
        batch_images_p = []
        batch_geometries_n = []
        batch_valid_mask_n = []
        batch_images_n = []
        batch_ones = []

        rand_state = np.random.get_state()
        if not self.shuffle:
            np.random.seed(idx)

        for element in training_plan:
            white_list = element[2]

            line_count, cluster_lines, cluster_label, cluster_class, cluster_images, cluster_id = \
                self.scenes[element[0]].get_cluster(element[1], self.max_line_count, self.shuffle, center=True)

            geometries_a, valid_mask_a, images_a = self.process_cluster(line_count, cluster_lines, cluster_label,
                                                                        cluster_class, cluster_images)

            line_count_p, lines_p, label_p, class_p, images_p, id_p = \
                self.scenes[element[0]].get_cluster(element[1], self.max_line_count, True,
                                                    blacklisted=cluster_id, center=True)
            assert(id_p != cluster_id)

            geometries_p, valid_mask_p, images_p = self.process_cluster(line_count_p, lines_p, label_p,
                                                                        class_p, images_p)

            other_scene_id = np.random.choice(white_list)
            other_scene = self.scenes[other_scene_id]
            # With probability 0.5, try to choose a cluster with the same class.
            if np.random.binomial(1, 0.5):
                same_classes_clusters = np.where(np.array(other_scene.cluster_classes) == cluster_class)[0]
                if len(same_classes_clusters) > 0:
                    other_cluster_id = np.random.choice(same_classes_clusters)
                else:
                    other_cluster_id = np.random.choice(np.arange(len(other_scene.cluster_labels)))
            else:
                other_cluster_id = np.random.choice(np.arange(len(other_scene.cluster_labels)))
            line_count_n, lines_n, label_n, class_n, images_n, id_n = \
                other_scene.get_cluster(other_cluster_id, self.max_line_count, True,
                                        center=True)

            geometries_n, valid_mask_n, images_n = self.process_cluster(line_count_n, lines_n, label_n,
                                                                        class_n, images_n)

            batch_geometries_a.append(geometries_a)
            batch_valid_mask_a.append(valid_mask_a)
            batch_images_a.append(images_a)
            batch_geometries_p.append(geometries_p)
            batch_valid_mask_p.append(valid_mask_p)
            batch_images_p.append(images_p)
            batch_geometries_n.append(geometries_n)
            batch_valid_mask_n.append(valid_mask_n)
            batch_images_n.append(images_n)
            batch_ones.append(np.expand_dims(np.ones((1, 1)), axis=0))

        geometries_a = np.concatenate(batch_geometries_a, axis=0)
        valid_mask_a = np.concatenate(batch_valid_mask_a, axis=0)
        images_a = np.concatenate(batch_images_a, axis=0)
        geometries_p = np.concatenate(batch_geometries_p, axis=0)
        valid_mask_p = np.concatenate(batch_valid_mask_p, axis=0)
        images_p = np.concatenate(batch_images_p, axis=0)
        geometries_n = np.concatenate(batch_geometries_n, axis=0)
        valid_mask_n = np.concatenate(batch_valid_mask_n, axis=0)
        images_n = np.concatenate(batch_images_n, axis=0)
        ones = np.concatenate(batch_ones, axis=0)

        if not self.shuffle:
            np.random.set_state(rand_state)

        return {'lines_a': geometries_a,
                'valid_input_mask_a': valid_mask_a,
                'images_a': images_a,
                'lines_p': geometries_p,
                'valid_input_mask_p': valid_mask_p,
                'images_p': images_p,
                'lines_n': geometries_n,
                'valid_input_mask_n': valid_mask_n,
                'images_n': images_n,
                'ones': ones}, ones

    def __len__(self):
        return int(np.ceil(self.cluster_count / self.batch_size))


def add_length(line_geometries):
    return np.hstack([line_geometries, np.linalg.norm(line_geometries[:, 3:6] - line_geometries[:, 0:3], axis=1)
                     .reshape((line_geometries.shape[0], 1))])


def subtract_mean(line_geometries, mean):
    # The mean is the mean of all start and end points.
    mean_vec = np.zeros((1, line_geometries.shape[1]))
    mean_vec[0, :3] = mean
    mean_vec[0, 3:6] = mean
    line_geometries = line_geometries - mean_vec

    return line_geometries


def normalize(line_geometries, std_dev):
    line_geometries[:, 0:6] = line_geometries[:, 0:6] / std_dev

    return line_geometries


def augment_global(line_geometries, angle_deviation, offset_deviation):
    # Rotates all lines in the scene around a random rotation vector.
    # Simulates slight viewpoint changes.

    theta = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, 2 * np.pi)
    x = np.sin(theta) * np.cos(psi)
    y = np.sin(theta) * np.sin(psi)
    z = np.cos(theta)
    angle = np.random.normal(0, angle_deviation)
    s = np.sin(angle)
    c = np.cos(angle)
    C = np.array([[x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                  [x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s],
                  [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c]])

    offset = np.random.normal([0, 0, 0], offset_deviation, (3,))

    # Rotate start points and end points
    line_geometries[:, :3] = np.transpose(C.dot(np.transpose(line_geometries[:, :3]))) + offset
    line_geometries[:, 3:6] = np.transpose(C.dot(np.transpose(line_geometries[:, 3:6]))) + offset

    # Rotate normals.
    line_geometries[:, 6:9] = np.transpose(C.dot(np.transpose(line_geometries[:, 6:9])))
    line_geometries[:, 9:12] = np.transpose(C.dot(np.transpose(line_geometries[:, 9:12])))


def augment_images(images):
    if np.random.binomial(1, 0.1):
        darkness = np.random.normal(1., 0.2)
        images[:, :, :, :] = images * darkness
    elif np.random.binomial(1, 0.1):
        images[:, :, :, :] = 0.
    elif np.random.binomial(1, 0.1):
        images[:, :, :, :] = np.random.normal(0., 30., images.shape)


def augment_flip(line_geometries, images):
    for i in range(line_geometries.shape[0]):
        if np.random.binomial(1, 0.5):
            buffer_start = np.copy(line_geometries[i, :3])
            line_geometries[i, :3] = line_geometries[i, 3:6]
            line_geometries[i, 3:6] = buffer_start
            buffer_normal_1 = np.copy(line_geometries[i, 6:9])
            line_geometries[i, 6:9] = line_geometries[i, 9:12]
            line_geometries[i, 9:12] = buffer_normal_1
            buffer_open_start = np.copy(line_geometries[i, 12])
            line_geometries[i, 12] = line_geometries[i, 13]
            line_geometries[i, 13] = buffer_open_start

            images[i, :, :, :] = np.flip(images[i, :, :, :], axis=2)

