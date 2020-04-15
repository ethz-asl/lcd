import numpy as np
import open3d as o3d
import pandas as pd
import os
import sys


class LineRenderer:
    def __init__(self, line_geometries, line_labels, label_colors):
        self.line_geometries = line_geometries
        self.line_labels = line_labels
        self.label_colors = label_colors
        self.render_normals = False
        self.indices = np.unique(line_labels)
        self.index_size = len(self.indices)
        self.pointer = 0
        self.render_single_cluster = False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        vis.register_key_callback(ord(" "), self.get_switch_index_callback())
        vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        vis.register_key_callback(ord("N"), self.get_toggle_normals_callback())

        vis.add_geometry(render_lines(self.line_geometries, self.label_colors))
        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        if self.render_single_cluster:
            vis.add_geometry(render_lines(
                self.line_geometries[np.where(self.line_labels == self.indices[self.pointer]), :][0],
                self.label_colors)
            )
        else:
            vis.add_geometry(render_lines(self.line_geometries, self.label_colors))

        if self.render_normals:
            if self.render_single_cluster:
                vis.add_geometry(render_normals(
                    self.line_geometries[np.where(self.line_labels == self.indices[self.pointer]), :][0])
                )
            else:
                vis.add_geometry(render_normals(self.line_geometries))
        vis.update_renderer()
        vis.get_view_control().convert_from_pinhole_camera_parameters(view)

    def get_toggle_normals_callback(self):
        def toggle_normals(vis):
            self.render_normals = not self.render_normals
            self.update_render(vis)

        return toggle_normals

    def get_toggle_show_all_callback(self):
        def toggle_show_all(vis):
            self.render_single_cluster = not self.render_single_cluster
            self.update_render(vis)

        return toggle_show_all

    def get_switch_index_callback(self):
        def switch_index(vis):
            if not self.render_single_cluster:
                self.render_single_cluster = True
            else:
                self.pointer = self.pointer + 1
                if self.pointer == self.index_size:
                    self.pointer = 0

            print("Now showing index {} ({}/{})".format(self.indices[self.pointer], self.pointer + 1, self.index_size))
            self.update_render(vis)

        return switch_index


def get_colors():
    colors = np.zeros((500, 3))

    for i in range(500):
        # Generate random numbers. With fixed seeds.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        colors[i, :] = rgb

    return colors


def render_lines(lines, label_colors):
    points = np.vstack((lines[:, 0:3], lines[:, 3:6]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    colors = [label_colors[int(lines[i, 6]), :].tolist() for i in range(line_count)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


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
    colors = [[0.8, 0.8, 0.8] for i in range(line_count*2)]
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


if __name__ == '__main__':
    lines = load_lines("/home/felix/line_ws/data/line_tools/interiornet_lines_split/all_lines_with_line_endpoints.txt")
    # For types: 7
    # For instances: 8
    # For classes: 9
    what_to_show = 'classes'
    if what_to_show == 'types':
        index = 7
    if what_to_show == 'instances':
        index = 8
    if what_to_show == 'classes':
        index = 9

    lines = lines[:, [1, 2, 3, 4, 5, 6, index, 10, 11, 12, 13, 14, 15, 16]]
    labels = lines[:, 6]

    print("Number of lines is: {}".format(lines.shape[0]))

    renderer = LineRenderer(lines, labels, get_colors())
    renderer.run()
