"""
Utility functions for sequential frame fusion.
These are currently not in use.
"""
from numba import jit, njit
from collections import Counter
import numpy as np


# This function is currently not in use.
def transform_lines(lines, transform):
    lines[:, :3] = transform.dot(np.vstack([lines[:, :3].T, np.ones((1, lines.shape[0]))]))[:3, :].T
    lines[:, 3:6] = transform.dot(np.vstack([lines[:, 3:6].T, np.ones((1, lines.shape[0]))]))[:3, :].T
    lines[:, 6:9] = transform[:3, :3].dot(lines[:, 6:9].T).T
    lines[:, 9:12] = transform[:3, :3].dot(lines[:, 9:12].T).T

    return lines


# This function is currently not in use.
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


# This function is currently not in use.
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


# This function is currently not in use.
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


# This function is currently not in use.
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


# This function is currently not in use.
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


# This function is currently not in use.
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