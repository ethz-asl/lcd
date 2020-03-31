import numpy as np
import open3d as o3d
import os
import pandas as pd
import sys


def get_colors():
    rgbs = np.zeros((500, 3))

    for i in range(500):
        # Generate random numbers. With fixed seeds.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        rgbs[i, :] = rgb

    return rgbs


def render_lines(lines, cluster_colors):
    points = np.vstack((lines[:, 0:3], lines[:, 3:6]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    print(lines.shape)
    print(cluster_colors.shape)
    colors = [cluster_colors[int(lines[i, 6]), :].tolist() for i in range(line_count)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def render_lines_index(lines, cluster_colors, index):
    return render_lines(lines[np.where(lines[:, 6] == index), :][0], cluster_colors)


def render_normals(lines):
    line_count = lines.shape[0]

    mid_points = (lines[:, 0:3] + lines[:, 3:6]) / 2
    ends_1 = mid_points + lines[:, 7:10] * 0.05
    ends_2 = mid_points + lines[:, 10:13] * 0.05

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
    except pd.errors.EmptyDataError:
        print("Error, empty data.")
    return data_lines


def switch_cluster(vis):
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    global index_, cluster_count_, indices_
    vis.add_geometry(render_lines_index(lines_, colors_, indices_[index_]))
    index_ = index_ + 1
    if index_ == cluster_count_:
        index_ = 0
    vis.update_renderer()
    vis.get_view_control().convert_from_pinhole_camera_parameters(view)
    return False


def show_all(vis):
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(render_lines(lines_, colors_))
    vis.get_view_control().convert_from_pinhole_camera_parameters(view)
    return False


def show_cluster(vis):
    switch_cluster(vis)

    global index_
    index_ = index_ - 1
    if index_ == -1:
        index_ = cluster_count_ - 1


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False


def render_clusters(lines):
    key_to_callback = {}
    key_to_callback[ord(" ")] = switch_cluster
    key_to_callback[ord("B")] = change_background_to_black
    key_to_callback[ord("A")] = show_all
    key_to_callback[ord("S")] = show_cluster

    global lines_, index_, cluster_count_, colors_, indices_
    colors_ = get_colors()
    lines_ = lines
    index_ = 0
    indices_ = np.unique(lines[:, 6])
    cluster_count_ = len(indices_)
    o3d.visualization.draw_geometries_with_key_callbacks([render_lines(lines, colors_)], key_to_callback)


if __name__ == '__main__':
    lines = load_lines("/home/felix/line_ws/data/line_tools/interiornet_lines_split/all_lines_with_line_endpoints.txt")
    lines = lines[:, [1,2,3,4,5,6,8,9,10,11,12,13,14,15]]
    #colors = get_colors()
    #o3d.visualization.draw_geometries([render_lines(lines, colors),
    #                                   render_normals(lines)])
    render_clusters(lines)
    print("Number of lines is: {}".format(lines.shape[0]))
