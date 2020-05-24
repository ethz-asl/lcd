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
    line_labels = data_lines.values[:, 15]
    line_class_ids = data_lines.values[:, 17]
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
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("WARNING: VIRTUAL CAMERA IMAGE NOT FOUND AT {}".format(path))
        return np.zeros(img_shape)
    else:
        img = preprocess_input(img)
        img = cv2.resize(img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img


class Scene:
    def __init__(self, path, bg_classes, min_line_count, max_clusters, img_shape):
        self.scene_path = path
        self.frame_paths = [os.path.join(path, name) for name in os.listdir(path)
                            if os.path.isfile(os.path.join(path, name))]
        self.frame_count = len(self.frame_paths)
        self.frames = [Frame(path) for path in self.frame_paths]
        self.bg_classes = bg_classes
        self.min_line_count = min_line_count
        self.max_clusters = max_clusters
        self.img_shape = img_shape
        self.valid_frames = []
        self.get_frames_info()

    def get_frames_info(self):
        line_counts = np.zeros((20,), dtype=int)
        cluster_counts = np.zeros((20,), dtype=int)
        scene_labels = np.zeros((0,), dtype=int)
        scene_classes = np.zeros((0,), dtype=int)

        # print("=============================================")

        for i, frame in enumerate(self.frame_paths):
            frame_idx = int(frame.split('_')[-1])
            count, geometries, labels, classes, vci_paths, transform = load_frame(frame)
            line_counts[frame_idx] = count
            clusters = Counter(labels[get_bg_mask(classes, self.bg_classes)]).most_common()
            cluster_counts[frame_idx] = len(clusters)
            scene_labels = np.hstack((scene_labels, labels))
            scene_classes = np.hstack((scene_classes, classes))
            # print("Frame line count: {}".format(line_counts[frame_idx]))
            # print("Frame cluster count: {}".format(cluster_counts[frame_idx]))
            cluster_line_counts = [cluster[1] for cluster in clusters]
            # print("Without clutter: {}".format(np.sum(np.where(np.array(cluster_line_counts) > 1, 1, 0))))
            # print(cluster_line_counts)

            if count > self.min_line_count and len(clusters) > 0:
                self.valid_frames.append(i)

        scene_clusters = Counter(scene_labels[get_bg_mask(scene_classes, self.bg_classes)]).most_common()
        scene_cluster_count = len(scene_clusters)

        # print("Scene: {}".format(self.scene_path))
        # print("Scene cluster count: {}".format(scene_cluster_count))
        # print("Scene line count: {}".format(np.sum(line_counts)))
        # print([cluster[1] for cluster in scene_clusters])

    def get_lines(self, frame_id, fusion_frames, line_count, shuffle):
        tot_count, lines, labels, classes, vci_paths, t_1 = load_frame(self.frame_paths[frame_id])
        vcis = [load_image(path, self.img_shape) for path in vci_paths]

        tic = time.perf_counter()
        for fuse_frame_id in fusion_frames:
            count_2, lines_2, labels_2, classes_2, vci_paths_2, t_2 = load_frame(self.frame_paths[fuse_frame_id])
            vcis_2 = [load_image(path, self.img_shape) for path in vci_paths_2]
            lines_2 = transform_lines(lines_2, np.linalg.inv(t_1).dot(t_2))
            lines, labels, classes, vcis = fuse_frames(np.array(lines), np.array(labels), np.array(classes), vcis,
                                                       lines_2, labels_2, classes_2, vcis_2)
            tot_count += count_2
            vcis = vcis + vcis_2
        count = len(lines)
        # print("Lines before fusing: {}".format(tot_count))
        # print("Lines after fusing: {}".format(count))
        # print("Fused {} lines.".format(tot_count - count))
        # print("Fusion took {} seconds.".format(time.perf_counter() - tic))

        labels = np.stack(labels)
        lines = np.vstack(lines)
        classes = np.stack(classes)

        # Delete lines of small clusters so that the number of clusters is below maximum.
        cluster_line_counts = Counter(np.array(labels)[get_bg_mask(classes, self.bg_classes)]).most_common()
        # print("Cluster count: {}".format(len(cluster_line_counts)))
        # print("Max cluster count: {}".format(self.max_clusters))
        for pair in cluster_line_counts[self.max_clusters:]:
            delete_idx = np.where(labels == pair[0])
            labels = np.delete(labels, delete_idx)
            lines = np.delete(lines, delete_idx, axis=0)
            classes = np.delete(classes, delete_idx)
            for i in np.sort(delete_idx[0])[::-1]:
                del(vcis[i])

        # print("Clutter lines deleted: {}".format(count - labels.shape[0]))
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


def get_bg_mask(classes, bg_classes):
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
                    # mask = np.logical_and(np.logical_and(img[:, :, 0] == 0, img[:, :, 1] == 0), img[:, :, 2] == 0)
                    img = preprocess_input(img)
                    # img[mask] = 0.0
                    img = cv2.resize(img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
                    images.append(np.expand_dims(img, axis=0))
            images = np.concatenate(images, axis=0)

        return count, self.line_geometries[indices, :], \
            self.line_labels[indices], self.line_class_ids[indices], images


def load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_clusters, img_shape):
    pickle_file_path = os.path.join(files_dir, "scenes_{}_{}_{}".format(min_line_count, max_line_count, max_clusters))
    if os.path.isfile(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        scene_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isdir(os.path.join(files_dir, name))]

        scenes = [Scene(path, bg_classes, min_line_count, max_clusters, img_shape) for path in scene_paths]
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
                 mean=np.zeros((3,)), img_shape=(64, 96, 3), min_line_count=30, max_line_count=300,
                 max_cluster_count=30, load_images=True):
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
        self.scenes = load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_cluster_count, img_shape)
        self.training_plan = create_training_plan(self.scenes)
        self.frame_count = len(self.training_plan)
        self.frame_indices = np.arange(self.frame_count)
        self.shuffle_indices()

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.frame_indices)

    def __getitem__(self, idx):
        training_plan = self.training_plan[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_geometries = []
        batch_labels = []
        batch_valid_mask = []
        batch_bg_mask = []
        batch_images = []
        batch_unique = []
        batch_counts = []

        for element in training_plan:
            if self.fuse:
                if not self.shuffle:
                    rand_state = np.random.get_state()
                    np.random.seed(element[0] * 20 + element[1])
                num_fuses = np.random.randint(4)
                other_frames = np.arange(self.scenes[element[0]].frame_count)
                other_frames = np.delete(other_frames, element[1])
                other_frames = np.random.choice(other_frames, num_fuses, False)
                if not self.shuffle:
                    np.random.set_state(rand_state)
            else:
                other_frames = []

            line_count, line_geometries, line_labels, line_class_ids, line_images = \
                self.scenes[element[0]].get_lines(element[1], other_frames, self.max_line_count, self.shuffle)

            if self.data_augmentation and np.random.binomial(1, 0.5):
                augment_flip(line_geometries, line_images)
                augment_global(line_geometries, np.radians(15.), 0.3)

            line_geometries = add_length(line_geometries)
            # line_geometries = subtract_mean(line_geometries, self.mean)
            # line_geometries = normalize(line_geometries, 2.0)

            out_valid_mask = np.zeros((self.max_line_count,), dtype=bool)
            out_geometries = np.zeros((self.max_line_count, line_geometries.shape[1]))
            out_labels = np.zeros((self.max_line_count,), dtype=int)
            out_classes = np.zeros((self.max_line_count,), dtype=int)
            out_bg = np.zeros((self.max_line_count,), dtype=bool)

            out_valid_mask[:line_count] = True
            out_geometries[:line_count, :] = line_geometries
            out_labels[:line_count] = line_labels
            out_classes[:line_count] = line_class_ids
            out_bg[np.isin(out_classes, self.bg_classes)] = True
            out_bg[line_count:self.max_line_count] = False

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

        return {'lines': geometries,
                'labels': labels,
                'valid_input_mask': valid_mask,
                'background_mask': bg_mask,
                'images': images,
                'unique_labels': unique,
                'cluster_count': counts}, labels

    def __len__(self):
        return int(np.ceil(self.frame_count / self.batch_size))


class LineDataGenerator:
    def __init__(self, files_dir, bg_classes,
                 shuffle=False, data_augmentation=False, mean=np.zeros((3,)), img_shape=(224, 224, 3),
                 sort=False, min_line_count=30, max_cluster_count=15):
        # Initialize parameters.
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.pointer = 0
        self.bg_classes = bg_classes
        self.frame_count = 0
        self.frames = []
        self.frame_indices = []
        self.mean = mean
        self.img_shape = img_shape
        self.sort = sort
        self.cluster_counts = []
        self.line_counts = []
        self.cluster_count_file = "cluster_counts"
        self.line_count_file = "line_counts"
        self.skipped_frames = 0
        self.min_line_count = min_line_count
        self.max_cluster_count = max_cluster_count
        self.load_scenes(files_dir)

        if self.shuffle:
            self.shuffle_data()

    def load_frames(self, files_dir):
        frame_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isfile(os.path.join(files_dir, name))]
        if not self.shuffle:
            frame_paths.sort()

        self.frame_count = len(frame_paths)
        self.frames = [Frame(path) for path in frame_paths]
        self.frame_indices = np.arange(self.frame_count)

    def load_scenes(self, files_dir):
        tic = time.perf_counter()
        scene_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isdir(os.path.join(files_dir, name))]
        if not self.shuffle:
            scene_paths.sort()

        self.frame_count = len(scene_paths)
        self.frames = [Scene(path, self.bg_classes, self.min_line_count, self.max_cluster_count, self.img_shape)
                       for path in scene_paths]
        self.frame_indices = np.arange(self.frame_count)
        print("Initialization took {} seconds.".format(time.perf_counter() - tic))

    def get_mean(self):
        mean = np.zeros((3,))
        count = 0
        for frame in self.frames:
            count = count + frame.line_count
            for i in range(frame.line_count):
                mean = mean + frame.line_geometries[i, :3] / 2.
                mean = mean + frame.line_geometries[i, 3:6] / 2.

        return mean / count

    def set_mean(self, mean):
        self.mean = mean

    def shuffle_data(self):
        """ Randomly shuffles the data stored.
        """
        np.random.shuffle(self.frame_indices)

    def reset_pointer(self):
        """ Resets internal pointer to point to the beginning of the stored
            data.
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

        print("Dataset completed. Number of skipped frames: {}".format(self.skipped_frames))
        self.skipped_frames = 0
        np.save(self.cluster_count_file, np.array(self.cluster_counts))
        np.save(self.line_count_file, np.array(self.line_counts))

    def set_pointer(self, index):
        """ Sets the internal pointer to point to the given index.

        Args:
            index (int): Index to which the internal pointer should point.
        """
        self.pointer = index

    def next_batch(self, batch_size, load_images):
        if self.pointer == self.frame_count:
            self.reset_pointer()

        line_count, line_geometries, line_labels, line_class_ids, line_images = \
            self.frames[self.frame_indices[self.pointer]].get_batch(batch_size,
                                                                    self.max_cluster_count,
                                                                    self.img_shape,
                                                                    self.shuffle,
                                                                    load_images)
        self.pointer = self.pointer + 1

        cluster_count = len(np.unique(line_labels[np.isin(line_class_ids, self.bg_classes, invert=True)]))

        # Write line counts and cluster counts for histogram:
        self.cluster_counts.append(cluster_count)
        self.line_counts.append(line_count)

        if line_count < self.min_line_count:
            # print("Skipping frame because it does not have enough lines")
            self.skipped_frames += 1
            return self.next_batch(batch_size, load_images)

        if cluster_count == 0 or cluster_count > self.max_cluster_count:
            # print("Skipping frame because it has 0 or too many instances.")
            self.skipped_frames += 1
            return self.next_batch(batch_size, load_images)

        out_k = np.zeros((31,))
        out_k[min(30, cluster_count)] = 1.

        # Subtract mean of start and end points.
        # Intuitively, the mean lies some where straight forward, i.e. [0., 0., 3.].
        line_geometries = subtract_mean(line_geometries, self.mean)
        line_geometries = normalize(line_geometries, 2.0)
        line_geometries = add_length(line_geometries)

        if self.data_augmentation and np.random.binomial(1, 0.5):
            augment_flip(line_geometries, line_images)
            augment_global(line_geometries, np.radians(15.), 0.2)

        # Sort by x value of leftest point:
        if self.sort:
            sorted_ids = np.argsort(np.min(line_geometries[:, [0, 3]], axis=1))
            line_geometries = line_geometries[sorted_ids, :]
            line_labels = line_labels[sorted_ids]
            line_class_ids = line_class_ids[sorted_ids]
            if load_images:
                line_images = line_images[sorted_ids, :, :, :]

        valid_mask = np.zeros((batch_size,), dtype=bool)
        out_geometries = np.zeros((batch_size, line_geometries.shape[1]))
        out_labels = np.zeros((batch_size,), dtype=int)
        out_classes = np.zeros((batch_size,), dtype=int)
        out_bg = np.zeros((batch_size,), dtype=bool)

        valid_mask[:line_count] = True
        out_geometries[:line_count, :] = line_geometries
        out_labels[:line_count] = line_labels
        out_classes[:line_count] = line_class_ids
        out_bg[np.isin(out_classes, self.bg_classes)] = True
        out_bg[line_count:batch_size] = False

        if load_images:
            # Images are sorted already.
            out_images = line_images
            if line_count < batch_size:
                out_images = np.concatenate([out_images,
                                             np.zeros((batch_size - line_count,
                                                       self.img_shape[0],
                                                       self.img_shape[1],
                                                       self.img_shape[2]))],
                                            axis=0)
        else:
            out_images = np.zeros((batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

        return out_geometries, out_labels, valid_mask, out_bg, out_images, out_k


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


def augment_local(line_geometries, offset_deviation, length_deviation, ):
    print("To be implemented.")


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


def kl_loss_np(prediction, labels, val_mask, bg_mask):
    h_labels = np.expand_dims(labels, axis=-1)
    v_labels = np.transpose(h_labels, axes=(1, 0))

    mask_equal = np.equal(h_labels, v_labels)
    mask_not_equal = np.logical_not(mask_equal)
    # mask_equal = np.expand_dims(mask_equal, axis=-1).astype(float)
    # mask_not_equal = np.expand_dims(mask_not_equal, axis=-1).astype(float)
    mask_equal = mask_equal.astype(float)
    mask_not_equal = mask_not_equal.astype(float)

    h_bg = np.expand_dims(np.logical_not(bg_mask), axis=-1).astype(float)
    v_bg = np.transpose(h_bg, axes=(1, 0))
    mask_not_bg_unexpanded = h_bg * v_bg
    # mask_not_bg = np.expand_dims(h_bg * v_bg, -1)
    mask_not_bg = mask_not_bg_unexpanded

    h_val = np.expand_dims(val_mask, axis=-1).astype(float)
    v_val = np.transpose(h_val, axes=(1, 0))
    mask_val = h_val * v_val
    mask_val_unexpanded = np.copy(mask_val)
    np.fill_diagonal(mask_val_unexpanded, 0.)
    # mask_val = np.expand_dims(mask_val_unexpanded, axis=-1)
    mask_val = mask_val_unexpanded

    h_pred = np.expand_dims(prediction, axis=0)
    v_pred = np.transpose(h_pred, axes=(1, 0, 2))

    d = h_pred * np.log(np.nan_to_num(h_pred / v_pred) + 0.000001)
    d = np.sum(d, axis=-1, keepdims=False)

    equal_loss_layer = mask_equal * d
    not_equal_loss_layer = mask_not_equal * np.maximum(0., 2.0 - d)
    # print(np.max(loss_layer * mask_val * mask_not_bg))
    # print("Compare:")
    # print(np.sum(not_equal_loss_layer * mask_val * mask_not_bg) + np.sum(equal_loss_layer * mask_val * mask_not_bg))


def get_fake_instancing(labels, valid_mask, bg_mask):
    valid_count = np.where(valid_mask == 1)[0].shape[0]
    unique_labels = np.unique(labels[np.where(np.logical_and(valid_mask, np.logical_not(bg_mask)))])
    out = np.zeros((150, 15), dtype=float)

    for i, label in enumerate(unique_labels):
        out[np.where(labels == label), i] = 1.

    kl_loss_np(out, labels, valid_mask, bg_mask)

    return np.expand_dims(out, axis=0)


# Deprecated
def generate_data(image_data_generator, max_line_count, line_num_attr, batch_size=1, load_images=True):
    batch_geometries = []
    batch_labels = []
    batch_valid_mask = []
    batch_bg_mask = []
    batch_images = []
    batch_k = []
    batch_fake = []
    batch_unique = []
    batch_counts = []

    for i in range(batch_size):
        # image_data_generator.reset_pointer()
        geometries, labels, valid_mask, bg_mask, images, k = image_data_generator.next_batch(max_line_count,
                                                                                             load_images=load_images)
        batch_labels.append(labels.reshape((1, max_line_count)))
        batch_geometries.append(geometries.reshape((1, max_line_count, line_num_attr)))
        batch_valid_mask.append(valid_mask.reshape((1, max_line_count)))
        batch_bg_mask.append(bg_mask.reshape((1, max_line_count)))
        batch_images.append(np.expand_dims(images, axis=0))
        batch_k.append(np.expand_dims(k, axis=0))
        batch_fake.append(np.zeros((1, max_line_count, 15)))

        unique_labels = np.unique(labels[np.logical_and(np.logical_not(bg_mask), valid_mask)])
        cluster_count = unique_labels.shape[0]
        if cluster_count >= 15:
            unique_labels = unique_labels[:15]
        else:
            unique_labels = np.pad(unique_labels, (0, 15 - cluster_count), mode='constant', constant_values=-1)
        unique_labels = np.expand_dims(unique_labels, axis=0)
        cluster_count = np.expand_dims(cluster_count, axis=0)
        batch_unique.append(unique_labels)
        batch_counts.append(cluster_count)

    geometries = np.concatenate(batch_geometries, axis=0)
    labels = np.concatenate(batch_labels, axis=0)
    valid_mask = np.concatenate(batch_valid_mask, axis=0)
    bg_mask = np.concatenate(batch_bg_mask, axis=0)
    images = np.concatenate(batch_images, axis=0)
    batch_k = np.concatenate(batch_k, axis=0)
    batch_fake = np.concatenate(batch_fake, axis=0)

    batch_unique = np.concatenate(batch_unique, axis=0)
    batch_counts = np.concatenate(batch_counts, axis=0)

    if not load_images:
        return {'lines': geometries,
                'labels': labels,
                'valid_input_mask': valid_mask,
                'background_mask': bg_mask,
                'unique_labels': batch_unique,
                'cluster_count': batch_counts,
                'fake': batch_fake}, batch_k
    else:
        return {'lines': geometries,
                'labels': labels,
                'valid_input_mask': valid_mask,
                'background_mask': bg_mask,
                'images': images,
                'unique_labels': batch_unique,
                'cluster_count': batch_counts,
                'fake': batch_fake}, batch_k


# Deprecated
def data_generator(image_data_generator, max_line_count, line_num_attr, batch_size=1, load_images=True):
    while True:
        yield generate_data(image_data_generator, max_line_count, line_num_attr, batch_size, load_images)
