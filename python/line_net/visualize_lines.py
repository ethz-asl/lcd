import numpy as np
import open3d as o3d
import pandas as pd
import os
import sys

from datagenerator_framewise import LineDataGenerator


class LineRenderer:
    def __init__(self, line_data_generator, results, pred_labels, margin, label_colors):
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
        self.pred_labels = pred_labels

        self.line_geometries, self.line_labels, self.valid_mask, self.bg_mask, images, k = \
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

        if self.render_result_connections and self.render_gt_connections:
            vis.add_geometry(render_lines(self.line_geometries,
                                          np.argmax(self.results[self.pointer, :, :], axis=-1), self.label_colors))
        else:
            vis.add_geometry(render_lines(self.line_geometries, self.line_labels, self.label_colors))

        if self.results is not None:
            result = self.results[self.pointer, :, :]
            # result = self.results[0, :, :]

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

                for i in range(150):
                    for j in range(150):
                        if self.bg_mask[i] or self.bg_mask[j] or not self.valid_mask[i] or not self.valid_mask[j]:
                            result_connections[i, j] = False

                print_metrics(result_connections, self.line_labels, self.bg_mask, self.valid_mask)
                print_iou(self.pred_labels[self.pointer, :], self.line_labels, self.bg_mask, self.valid_mask)

        if self.render_result_connections and self.render_gt_connections:
            # vis.add_geometry(render_compare(self.line_geometries, self.line_labels, self.bg_mask, result_connections))
            print("Currently not rendering")
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
            self.line_geometries, self.line_labels, self.valid_mask, self.bg_mask, images, k = \
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


def print_metrics(predictions, gt, bg, valid):
    v_bg = np.expand_dims(bg, axis=-1)
    h_bg = np.transpose(v_bg)
    not_bg_mask = np.logical_not(np.logical_and(v_bg, h_bg))

    v_valid = np.expand_dims(valid, axis=-1)
    h_valid = np.transpose(v_valid)
    valid_mask = np.logical_and(v_valid, h_valid)

    loss_mask = np.logical_and(not_bg_mask, valid_mask)

    v_gt = np.expand_dims(gt, axis=-1)
    h_gt = np.transpose(v_gt, axes=(1, 0))
    gt_equals = np.logical_and(np.equal(v_gt, h_gt), loss_mask)

    true_p = np.sum(np.logical_and(predictions, gt_equals).astype(float))
    false_p = np.sum(np.logical_and(predictions, np.logical_not(gt_equals))).astype(float)
    false_n = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(gt_equals))).astype(float)
    gt_p = np.sum(gt_equals.astype(float))
    pred_p = np.sum(predictions.astype(float))

    print("tp_gt_p: {}".format(true_p / gt_p))
    print("tp_pd_p: {}".format(true_p / pred_p))


def print_iou(pred_labels, gt, bg, valid):
    mask = np.logical_and(np.logical_not(bg), valid)

    labels = gt[mask]
    unique_labels = np.unique(labels)
    cluster_count = unique_labels.shape[0]
    unique_labels = np.pad(unique_labels, (0, 15 - cluster_count), mode='constant', constant_values=0)
    pred_labels = pred_labels[mask]

    gt_labels = np.expand_dims(np.expand_dims(labels, axis=-1), axis=-1)
    unique_gt_labels = np.expand_dims(np.expand_dims(unique_labels, axis=0), axis=-1)
    pred_labels = np.expand_dims(np.expand_dims(pred_labels, axis=-1), axis=-1)
    unique_pred_labels = np.expand_dims(np.expand_dims(np.arange(0, 15), axis=0), axis=0)

    gt_matrix = np.equal(unique_gt_labels, gt_labels)
    pred_matrix = np.equal(unique_pred_labels, pred_labels)

    intersections = np.logical_and(gt_matrix, pred_matrix)
    unions = np.logical_or(gt_matrix, pred_matrix)
    intersections = np.sum(intersections, axis=0)
    unions = np.sum(unions, axis=0)

    ious = np.max(np.nan_to_num(intersections / unions), axis=1)
    iou = np.sum(ious) / cluster_count

    print("IoU: {}".format(iou))


def load_lines(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
    except pd.errors.EmptyDataError:
        print("Error, empty data.")
    return data_lines


if __name__ == '__main__':
    data_path = "/nvme/line_ws/test"
    result_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/120520_2010/results"

    data_generator = LineDataGenerator(data_path,
                                       [0, 1, 2, 20, 22],
                                       sort=True,
                                       min_line_count=0,
                                       max_cluster_count=1000000)

    predictions = np.load(os.path.join(result_path, "predictions_train.npy"))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=-1)
    h_pred = np.expand_dims(predictions, axis=-1)
    v_pred = np.transpose(h_pred, axes=(0, 2, 1))
    results = np.equal(h_pred, v_pred).astype(float)
    print(predictions[0, :])

    renderer = LineRenderer(data_generator, results, predictions, 0.7, get_colors())
    renderer.run()
