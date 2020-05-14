import os
import numpy as np

from tensorflow.keras.models import load_model

from datagenerator_framewise import LineDataGenerator

def infer():
    test_files = "/nvme/line_ws/test"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (120, 180, 3)
    max_line_count = 150
    batch_size = 20
    num_epochs = 50
    bg_classes = [0, 1, 2, 20, 22]

    log_dir = "/home/felix/line_ws/src/line_tools/python/line_net/logs/130520_2315"
    epoch = 47

    model = load_model(os.path.join(log_dir, "weights.{}.hdf5".format(epoch)))

    predictions = []
    gts = []

    test_data_generator = LineDataGenerator(test_files, bg_classes,
                                            mean=train_set_mean, img_shape=img_shape, sort=True,
                                            min_line_count=0, max_cluster_count=100000)
    for i in range(test_data_generator.frame_count):
        geometries, labels, valid_mask, bg_mask, images, k = \
            test_data_generator.next_batch(max_line_count, load_images=False)

        fake = np.zeros((1, max_line_count, 15))
        labels = labels.reshape((1, max_line_count))
        geometries = geometries.reshape((1, max_line_count, line_num_attr))
        valid_mask = valid_mask.reshape((1, max_line_count))
        bg_mask = bg_mask.reshape((1, max_line_count))
        images = np.expand_dims(images, axis=0)

        output = line_model.predict({'lines': geometries,
                                     'labels': labels,
                                     'valid_input_mask': valid_mask,
                                     'background_mask': bg_mask,
                                     # 'images': images,
                                     'fake': fake})

        predictions.append(output)
        gts.append(labels)

        print("Frame {}/{}".format(i, test_data_generator.frame_count))
        # np.save("output/output_frame_{}".format(i), output)

    results_path = os.path.join(log_path, "results")
    os.mkdir(results_path)
    np.save(os.path.join(results_path, "predictions_train"), np.array(predictions))
    np.save(os.path.join(results_path, "ground_truths_train"), np.array(gts))


if __name__ == '__main__':