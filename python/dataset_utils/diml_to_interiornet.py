import cv2
import os
import shutil
import numpy as np


def sample_to_interiornet():
    diml_path = "/nvme/datasets/diml_depth/scenes"
    hd7_path = "/nvme/datasets/diml_depth/HD7"

    depth_paths = [
        "/nvme/datasets/diml_depth/train/HR/11. Bedroom/depth_filled",
        "/nvme/datasets/diml_depth/train/HR/12. Livingroom/depth_filled"]

    depth_images = []
    for path in depth_paths:
        depth_images += [os.path.join(path, name) for name in os.listdir(path)
                         if os.path.isfile(os.path.join(path, name))]

    scene_paths = [os.path.join(diml_path, name) for name in os.listdir(diml_path)
                   if os.path.isdir(os.path.join(diml_path, name))]

    for scene_path in scene_paths:
        frame_paths = [os.path.join(scene_path, name) for name in os.listdir(scene_path)
                       if os.path.isfile(os.path.join(scene_path, name))]

        new_frame_path = os.path.join(hd7_path, scene_path.split('/')[-1])
        os.mkdir(new_frame_path)
        os.mkdir(os.path.join(new_frame_path, "cam0"))
        os.mkdir(os.path.join(new_frame_path, "depth0"))
        os.mkdir(os.path.join(new_frame_path, "label0"))
        os.mkdir(os.path.join(new_frame_path, "cam0", "data"))
        os.mkdir(os.path.join(new_frame_path, "depth0", "data"))
        os.mkdir(os.path.join(new_frame_path, "label0", "data"))
        print(new_frame_path)
        for i, frame_path in enumerate(frame_paths):
            file_name = frame_path.split('/')[-1][:-6]
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            print(file_name)
            depth_path = [path for path in depth_images if file_name in path][0]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            img = cv2.resize(img, dsize=(img.shape[1] / 2, img.shape[0] / 2), interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, dsize=(depth_img.shape[1] / 2, depth_img.shape[0] / 2),
                                   interpolation=cv2.INTER_LINEAR)
            label_img = depth_img.copy()
            label_img[:, :] = 3

            cv2.imwrite(os.path.join(new_frame_path, "cam0", "data", "{}.png".format(i)), img)
            cv2.imwrite(os.path.join(new_frame_path, "depth0", "data", "{}.png".format(i)), depth_img)
            cv2.imwrite(os.path.join(new_frame_path, "label0", "data", "{}_instance.png".format(i)), label_img)
            cv2.imwrite(os.path.join(new_frame_path, "label0", "data", "{}_nyu.png".format(i)), label_img)


def full_to_interiornet():
    scene_file_path = "/nvme/datasets/diml_depth/scenes.txt"
    base_path = "/nvme/datasets/diml_depth/"
    out_path = "/nvme/datasets/diml_depth/HD7/"
    cam0_render = "/nvme/datasets/interiornet/3FO4IDEI1LAV_Bedroom/cam0.render"
    num_frames = 20
    shape = (672, 378)
    np.random.seed(123)

    with open(scene_file_path, 'r') as f:
        scene_lines = f.readlines()

    scene_lines = [sn.split('\n')[0] for sn in scene_lines]
    scene_paths = [os.path.join(base_path, sn.split('-')[0]) for sn in scene_lines]
    scene_ranges = [sn.split('-')[1] for sn in scene_lines]
    scene_ranges = [(int(rn[1:-1].split(':')[0]), int(rn[1:-1].split(':')[1])) for rn in scene_ranges]

    for i, scene_path in enumerate(scene_paths):
        file_list = []
        for j in range(scene_ranges[i][0], scene_ranges[i][1]+1):
            scene_path_col = os.path.join(scene_path, "{}/col".format(j))
            if os.path.exists(scene_path_col):
                file_list += [os.path.join(scene_path_col, dn) for dn in os.listdir(scene_path_col)]

        scene_count = len(os.listdir(out_path))
        scene_out_path = "{:02d}DIML_{}".format(scene_count + 1, scene_path.split('/')[-2].split(' ')[1])
        scene_out_path = os.path.join(out_path, scene_out_path)

        if os.path.exists(scene_out_path):
            shutil.rmtree(scene_out_path)
        os.mkdir(scene_out_path)
        os.mkdir(os.path.join(scene_out_path, "cam0"))
        os.mkdir(os.path.join(scene_out_path, "depth0"))
        os.mkdir(os.path.join(scene_out_path, "label0"))
        os.mkdir(os.path.join(scene_out_path, "cam0", "data"))
        os.mkdir(os.path.join(scene_out_path, "depth0", "data"))
        os.mkdir(os.path.join(scene_out_path, "label0", "data"))
        shutil.copyfile(cam0_render, os.path.join(scene_out_path, "cam0.render"))
        print(scene_out_path)

        frame_paths = np.random.choice(file_list, num_frames, False)
        for j, frame_path in enumerate(frame_paths):
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            depth_path = frame_path.replace('/col/', '/up_png/')
            depth_path = depth_path.replace('_c.png', '_ud.png')
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if depth_img is None:
                print(depth_path)
                exit()
            if img is None:
                print(frame_path)
                exit()

            img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, dsize=shape,
                                   interpolation=cv2.INTER_LINEAR)
            label_img = depth_img.copy()
            label_img[:, :] = 3

            cv2.imwrite(os.path.join(scene_out_path, "cam0", "data", "{}.png".format(j)), img)
            cv2.imwrite(os.path.join(scene_out_path, "depth0", "data", "{}.png".format(j)), depth_img)
            cv2.imwrite(os.path.join(scene_out_path, "label0", "data", "{}_instance.png".format(j)), label_img)
            cv2.imwrite(os.path.join(scene_out_path, "label0", "data", "{}_nyu.png".format(j)), label_img)


if __name__ == '__main__':
    full_to_interiornet()
