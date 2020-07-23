import shutil

import cv2
import hdf5storage as hdf
import numpy as np
import pickle
import os


def get_cam0(num_frames):
    out = "#first three elements, eye and next three, lookAt and the last there, up direction \n" +\
          "#frame rate per second: 0 \n" +\
          "#shutter_speed (ns): 0"
    for i in range(num_frames):
        out += "\n{} 2.10833 1.15056 1.38964 1.33815 1.77933 1.28258 2.09727 1.30522 2.37755".format(i)
        out += "\n{} 2.10833 1.15056 1.38964 1.33815 1.77933 1.28258 2.09727 1.30522 2.37755".format(i)

    return out


def get_instances(labels, instances):
    out_img = instances.copy()
    unique_classes = np.unique(labels)
    num_instances = 0
    for unique_class in unique_classes:
        if unique_class == 0:
            # Background is 0.
            continue
        unique_instances = np.unique(instances[labels == unique_class])
        for unique_instance in unique_instances:
            out_img[np.logical_and(labels == unique_class, instances == unique_instance)] = num_instances
            num_instances += 1

    return out_img


if __name__ == '__main__':
    out_path = "/nvme/datasets/nyu_v2/HD7"
    cam0_render = "/nvme/datasets/interiornet/3FO4IDEI1LAV_Bedroom/cam0.render"

    dataset = hdf.loadmat("/nvme/nyu_depth_v2_labeled.mat")

    scene_names = []
    frame_counts = []

    for scene in dataset['scenes']:
        scene_name = scene[0][0][0]

        if scene_name not in scene_names:
            scene_names.append(scene_name)
            frame_counts.append(1)
        else:
            frame_counts[scene_names.index(scene_name)] += 1

    print(scene_names)
    print(frame_counts)
    print("Max frame count: {}".format(max(frame_counts)))
    print("Min frame count: {}".format(min(frame_counts)))
    print("Number of scenes: {}".format(len(frame_counts)))
    print("Number of scenes with more than 1 frame: {}".format(np.sum(np.where(np.array(frame_counts) > 1, 1, 0))))
    print("Number of scenes with more than 3 frames: {}".format(np.sum(np.where(np.array(frame_counts) > 3, 1, 0))))

    for i, scene_name in enumerate(scene_names):
        if frame_counts[i] < 1:
            continue

        scene_out_path = os.path.join(out_path, "{:03d}_".format(i) + scene_name)

        if os.path.exists(scene_out_path):
            shutil.rmtree(scene_out_path)
        os.mkdir(scene_out_path)
        os.mkdir(os.path.join(scene_out_path, "cam0"))
        os.mkdir(os.path.join(scene_out_path, "depth0"))
        os.mkdir(os.path.join(scene_out_path, "label0"))
        os.mkdir(os.path.join(scene_out_path, "cam0", "data"))
        os.mkdir(os.path.join(scene_out_path, "depth0", "data"))
        os.mkdir(os.path.join(scene_out_path, "label0", "data"))
        with open(os.path.join(scene_out_path, "cam0.render"), 'w') as f:
            f.write(get_cam0(frame_counts[i]))
        print(scene_out_path)

        frame_id = 0
        for j, other_scene in enumerate(dataset['scenes']):
            other_scene_name = other_scene[0][0][0]

            if other_scene_name != scene_name:
                continue

            crop = 8
            img = dataset['images'][crop:-crop, crop:-crop, :, j]
            img = np.flip(img, axis=-1)
            depth_img = dataset['depths'][crop:-crop, crop:-crop, j]
            depth_img = (depth_img * 1000).astype(np.uint16)
            label_img = dataset['labels'][crop:-crop, crop:-crop, j].astype(np.uint16)
            instance_img = dataset['instances'][crop:-crop, crop:-crop, j].astype(np.uint16)

            instance_processed = get_instances(label_img, instance_img)

            cv2.imwrite(os.path.join(scene_out_path, "cam0", "data", "{}.png".format(frame_id)), img)
            cv2.imwrite(os.path.join(scene_out_path, "depth0", "data", "{}.png".format(frame_id)), depth_img)
            cv2.imwrite(os.path.join(scene_out_path, "label0", "data", "{}_instance.png".format(frame_id)), instance_processed)
            cv2.imwrite(os.path.join(scene_out_path, "label0", "data", "{}_nyu.png".format(frame_id)), label_img)

            frame_id += 1

