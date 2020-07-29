"""
Renderer of the inference results from training to view the clustering performance of the neural network for
clustering.
"""
import numpy as np
import open3d as o3d
import os
import argparse


class LineRenderer:
    def __init__(self, predictions, gt_labels, geometries, backgrounds, valids, label_colors):
        self.predictions = predictions
        self.geometries = geometries
        self.gt_labels = gt_labels
        self.backgrounds = backgrounds
        self.valids = valids
        self.label_colors = label_colors
        self.indices = np.arange(predictions.shape[0])
        self.index_size = len(self.indices)
        self.pointer = 0
        self.render_single_cluster = False
        self.render_gt = False
        self.render_normals = False
        self.render_result_connections = False
        self.get_data()

    def get_data(self):
        self.line_geometries = self.geometries[self.pointer, :, :].squeeze()
        self.line_labels = self.gt_labels[self.pointer, :].squeeze()
        self.valid_mask = self.valids[self.pointer, :].squeeze()
        self.bg_mask = self.backgrounds[self.pointer, :].squeeze()
        print("Number of lines in scene: {}".format(np.sum(self.valid_mask)))

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        print(opt.line_width)
        opt.line_width = 20.
        ctr = vis.get_view_control()
        ctr.rotate(1100.0, 0.0)

        vis.register_key_callback(ord("N"), self.get_toggle_normals_callback())
        vis.register_key_callback(ord("E"), self.get_set_ground_truth_callback())
        vis.register_key_callback(ord("Q"), self.get_set_prediction_callback())

        print("Press 'N' to show normals.")
        print("Press 'E' to show ground truth labels.")
        print("Press 'Q' to show predicted labels.")

        self.update_render(vis)

        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        opt = vis.get_render_option()
        opt.line_width = 20.

        pred_labels = self.predictions[self.pointer, :].squeeze()
        pred_bg = np.where(pred_labels == 0, True, False)
        if not self.render_gt:
            print("Rendering predictions.")
            vis.add_geometry(render_lines(self.line_geometries, pred_labels, pred_bg, self.label_colors))
        else:
            print("Rendering ground truth.")
            vis.add_geometry(render_lines(self.line_geometries, self.line_labels, self.bg_mask, self.label_colors))

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

    def get_set_ground_truth_callback(self):
        def toggle_ground_truth(vis):
            self.render_gt = not True
            self.update_render(vis)

        return toggle_ground_truth

    def get_set_prediction_callback(self):
        def toggle_prediction(vis):
            self.render_gt = not False
            self.update_render(vis)

        return toggle_prediction

    # Currently not in use.
    def get_fuse_callback(self, do_fuse):
        def fuse(vis):
            self.line_data_generator.fuse = not self.line_data_generator.fuse
            self.get_data()
            self.update_render(vis)
        return fuse

    def get_switch_index_callback(self):
        def switch_index(vis):
            self.pointer = (self.pointer + 1) % self.index_size
            self.get_data()

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

    colors = np.vstack([np.array([
        [255, 0, 0],
        [128, 64, 64],
        [255, 128, 0],
        [255, 255, 0],
        [128, 255, 0],
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
    ]) / 255., colors])

    return colors


def render_lines(lines, labels, bg_mask, label_colors):
    bg_color = np.array([1.0, 0.0, 0.0])

    points = np.vstack((lines[:, 0:3], lines[:, 3:6]))
    line_count = lines.shape[0]
    indices = np.hstack((np.arange(line_count).reshape(line_count, 1),
                         (np.arange(line_count) + line_count).reshape(line_count, 1)))

    colors = [label_colors[int(labels[i]), :].tolist() for i in range(line_count)]
    for i in range(line_count):
        if bg_mask[i]:
            colors[i] = bg_color
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Render the line clusterings from the inference during training of the clustering network.')
    parser.add_argument(
        "--path_to_results",
        default=None,
        help="Path to where the the inference results are saved to.")
    args = parser.parse_args()

    result_path = args.path_to_results

    predictions = np.load(os.path.join(result_path, "predictions.npy"))
    predictions = np.argmax(predictions[:, :, 0:], axis=-1)
    gt_labels = np.load(os.path.join(result_path, "labels.npy"))
    geometries = np.load(os.path.join(result_path, "geometries.npy"))
    backgrounds = np.load(os.path.join(result_path, "backgrounds.npy"))
    valids = np.load(os.path.join(result_path, "valids.npy"))

    renderer = LineRenderer(predictions, gt_labels, geometries, backgrounds, valids, get_colors())
    renderer.run()
