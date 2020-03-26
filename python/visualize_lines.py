import numpy as np
import open3d as o3d
import os
import pandas as pd
import sys


def render_lines(lines):
    points = np.vstack((lines[:, 1:4], lines[:, 4:7]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    colors = [[0, 0.2, 0.8] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])


def load_lines(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
        line_count = data_lines.shape[0]
    except pd.errors.EmptyDataError:
        line_count = 0
    return data_lines


if __name__ == '__main__':
    lines = load_lines("/home/felix/line_ws/data/line_tools/interiornet_lines_split/all_lines_with_line_endpoints.txt")
    render_lines(lines)
