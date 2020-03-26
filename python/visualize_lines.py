import numpy as np
import open3d as o3d
import os
import pandas as pd
import sys


def get_colors():
    interpolate = np.linspace(0, 1, 100)

    rgbs = np.zeros((500, 3))

    for i in range(500):
        # NOTE: Here it is crucial to note that when initializing the random
        # number generator everytime immediately before generating the
        # random number, the latter will always be same if the same seed is
        # used. Therefore, same lines_color[i] will always correspond to the
        # same colour.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        rgbs[i, :] = rgb

    return rgbs


def render_lines(lines, cluster_colors):
    points = np.vstack((lines[:, 1:4], lines[:, 4:7]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    colors = [cluster_colors[int(lines[i, 8]), :].tolist() for i in range(line_count)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def render_normals(lines):
    line_count = lines.shape[0]

    mid_points = (lines[:, 1:4] + lines[:, 4:7]) / 2
    ends_1 = mid_points + lines[:, 9:12] * 0.05
    ends_2 = mid_points + lines[:, 12:15] * 0.05

    points = np.vstack((mid_points, ends_1, ends_2))
    indices = np.vstack((
                np.hstack((np.arange(line_count).reshape(line_count, 1),
                    (np.arange(line_count) + line_count).reshape(line_count, 1))),
                np.hstack((np.arange(line_count).reshape(line_count, 1),
                    (np.arange(line_count) + line_count*2).reshape(line_count, 1)))
    ))
    colors = [[0, 0, 0] for i in range(line_count)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def load_lines(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
        line_count = data_lines.shape[0]
    except pd.errors.EmptyDataError:
        line_count = 0
    return data_lines


if __name__ == '__main__':
    lines = load_lines("/home/felix/line_ws/data/line_tools/interiornet_lines_split/train_with_line_endpoints.txt")
    colors = get_colors()
    o3d.visualization.draw_geometries([render_lines(lines, colors),
                                       render_normals(lines)])
