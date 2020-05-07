import numpy as np
import open3d as o3d
import pandas as pd
import os
import sys

from datagenerator_framewise import LineDataGenerator


class LineRenderer:
    def __init__(self, line_data_generator, results, margin, label_colors):
        self.line_data_generator = line_data_generator
        self.results = results
        self.label_colors = label_colors
        self.margin = margin
        self.indices = np.arange(line_data_generator.frame_count)
        self.index_size = len(self.indices)
        self.pointer = 0
        self.render_single_cluster = False
        self.render_gt_connections = False
        self.render_normals = False
        self.render_result_connections = False
        self.show_closest = False

        self.line_geometries, self.line_labels, self.valid_mask, self.bg_mask, images = \
            self.line_data_generator.next_batch(150, False)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        ctr = vis.get_view_control()
        ctr.rotate(1100.0, 0.0)

        vis.register_key_callback(ord(" "), self.get_switch_index_callback())
        vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        vis.register_key_callback(ord("N"), self.get_toggle_normals_callback())
        vis.register_key_callback(ord("E"), self.get_toggle_show_connections_callback())
        vis.register_key_callback(ord("Q"), self.get_toggle_show_results_callback())
        vis.register_key_callback(ord("C"), self.get_toggle_show_closest_callback())
        vis.register_key_callback(ord("M"), self.get_increase_margin_callback())
        vis.register_key_callback(ord("N"), self.get_decrease_margin_callback())

        print("Press space to switch label index.")
        print("Press 'A' to toggle between show all and show instance.")
        print("Press 'N' to show normals.")
        print("Press 'E' to show ground truth connections.")
        print("Press 'Q' to show predicted connections.")
        print("Press 'C' to toggle between show closest and show margin.")
        print("Press 'M' to increase margin.")
        print("Press 'N' to decrease margin.")

        self.update_render(vis)

        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        # indices = np.where(self.line_labels == self.indices[self.pointer])

        # if self.render_single_cluster:
        #     vis.add_geometry(render_lines(
        #         self.line_geometries[indices, :][0],
        #         self.line_labels[indices],
        #         self.label_colors)
        #     )
        # else:
        #     vis.add_geometry(render_lines(self.line_geometries, self.line_labels, self.label_colors))

        # if self.render_normals:
        #     if self.render_single_cluster:
        #         vis.add_geometry(render_normals(
        #             self.line_geometries[indices, :][0])
        #         )
        #     else:
        #         vis.add_geometry(render_normals(self.line_geometries))

        vis.add_geometry(render_lines(self.line_geometries, self.line_labels, self.label_colors))

        result = self.results[self.pointer]
        #
        if self.show_closest:
            max_results = np.zeros_like(result, dtype=bool)
            max_result_sort = np.argsort(result, axis=-1)[:, -1:]
            for i in range(result.shape[0]):
                max_results[i, max_result_sort[i, :]] = True
            max_results[:, np.logical_not(self.valid_mask)] = False
            max_results[np.logical_not(self.valid_mask)] = False
            # max_results[np.where(result < self.margin)] = False
            result_connections = max_results
        else:
            result_connections = np.where(result > self.margin, True, False)

        if self.render_result_connections and self.render_gt_connections:
            vis.add_geometry(render_compare(self.line_geometries, self.line_labels, self.bg_mask, result_connections))
        elif self.render_gt_connections:
            vis.add_geometry(render_gt_connections(self.line_geometries, self.line_labels, self.bg_mask))
        elif self.render_result_connections:
            vis.add_geometry(render_connections(self.line_geometries, result_connections))

        if self.render_normals:
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

    def get_increase_margin_callback(self):
        def increase_margin(vis):
            self.margin = self.margin + 0.05
            self.update_render(vis)

        return increase_margin

    def get_decrease_margin_callback(self):
        def decrease_margin(vis):
            self.margin = self.margin - 0.05
            self.update_render(vis)

        return decrease_margin

    def get_toggle_show_connections_callback(self):
        def toggle_show_connections(vis):
            self.render_gt_connections = not self.render_gt_connections
            self.update_render(vis)

        return toggle_show_connections

    def get_toggle_show_results_callback(self):
        def toggle_show_results(vis):
            self.render_result_connections = not self.render_result_connections
            self.update_render(vis)

        return toggle_show_results

    def get_toggle_show_closest_callback(self):
        def toggle_show_closest(vis):
            self.show_closest = not self.show_closest
            self.update_render(vis)

        return toggle_show_closest

    def get_switch_index_callback(self):
        def switch_index(vis):
            # if not self.render_single_cluster:
            #     self.render_single_cluster = True
            # else:
            self.line_geometries, self.line_labels, self.valid_mask, self.bg_mask, images = \
                self.line_data_generator.next_batch(150, load_images=False)
            self.pointer = (self.pointer + 1) % self.index_size

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


def render_lines(lines, labels, label_colors):
    points = np.vstack((lines[:, 0:3], lines[:, 3:6]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    colors = [label_colors[int(labels[i]), :].tolist() for i in range(line_count)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def render_normals(lines):
    line_count = lines.shape[0]

    mid_points = (lines[:, 0:3] + lines[:, 3:6]) / 2.
    ends_1 = mid_points + lines[:, 6:9] * 0.05
    ends_2 = mid_points + lines[:, 9:12] * 0.05

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


def render_compare(lines, labels, bg_mask, predictions):
    labels_h = np.expand_dims(labels, -1)
    labels_h = np.repeat(labels_h, labels.shape[0], axis=1)
    labels_v = np.transpose(labels_h)
    gt_connections = np.equal(labels_h, labels_v)
    gt_connections[bg_mask, :] = False
    gt_connections[:, bg_mask] = False

    mid_points = (lines[:, 0:3] + lines[:, 3:6]) / 2.
    indices = []
    colors = []
    for i in range(lines.shape[0]):
        for j in range(lines.shape[0]):
            if not i == j:
                if gt_connections[i, j] and predictions[i, j]:
                    # Correct prediction:
                    indices.append(np.array([i, j]))
                    colors.append([0.8, 0.8, 0.8])
                elif gt_connections[i, j]:
                    # Missing prediction:
                    indices.append(np.array([i, j]))
                    colors.append([0.3, 0.3, 0.3])
                elif predictions[i, j]:
                    # Wrong prediction:
                    indices.append(np.array([i, j]))
                    colors.append([0.8, 0., 0.])

    indices = np.array(indices, dtype=int)

    # colors = [[0.8, 0.8, 0.8] for i in range(len(indices))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mid_points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def render_gt_connections(lines, labels, bg_mask):
    labels_h = np.expand_dims(labels, -1)
    labels_h = np.repeat(labels_h, labels.shape[0], axis=1)
    labels_v = np.transpose(labels_h)
    connections = np.equal(labels_h, labels_v)
    connections[bg_mask, :] = False
    connections[:, bg_mask] = False

    return render_connections(lines, connections)


def render_connections(lines, connections):
    mid_points = (lines[:, 0:3] + lines[:, 3:6]) / 2.
    indices = []
    for i in range(lines.shape[0]):
        for j in range(lines.shape[0]):
            if not i == j:
                if connections[i, j]:
                    indices.append(np.array([i, j]))
    indices = np.array(indices, dtype=int)

    colors = [[0.8, 0.8, 0.8] for i in range(len(indices))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mid_points.astype(float).tolist()),
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
    data_path = "/nvme/line_ws/train_data/val"
    result_path = "output"
    results = []

    data_generator = LineDataGenerator(data_path,
                                       [0, 1, 2, 20, 22])

    for i in range(10):
        results.append(np.load(os.path.join(result_path, "output_frame_{}.npy".format(i))))

    renderer = LineRenderer(data_generator, results, 0.7, get_colors())
    renderer.run()
