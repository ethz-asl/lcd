import os
import numpy as np

from tensorflow.keras.models import load_model

from datagenerator_framewise import LineDataGenerator
from datagenerator_framewise import generate_data
from model import line_net_model_3


def infer():
    test_files = "/nvme/line_ws/test"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (120, 180, 3)
    max_line_count = 150
    bg_classes = [0, 1, 2, 20, 22]

    log_dir = "/home/felix/line_ws/src/line_tools/python/line_net/logs/130520_2315"
    epoch = 47

    model = line_net_model_3(line_num_attr, max_line_count, img_shape)
    model.load_weights(os.path.join(log_dir, "weights.{}.hdf5".format(epoch)))

    predictions = []
    gts = []

    train_set_mean = np.array([0.0, 0.0,  3.15564408])
    test_data_generator = LineDataGenerator(test_files, bg_classes,
                                            mean=train_set_mean, img_shape=img_shape, sort=True,
                                            min_line_count=0, max_cluster_count=100000)
    for i in range(test_data_generator.frame_count):
        data, gt = generate_data(test_data_generator, max_line_count, line_num_attr, 1)

        output = model.predict(data)

        predictions.append(output)
        gts.append(data['labels'])

        print("Frame {}/{}".format(i, test_data_generator.frame_count))
        # np.save("output/output_frame_{}".format(i), output)

    results_path = os.path.join(log_dir, "results")
    os.mkdir(results_path)
    np.save(os.path.join(results_path, "predictions_train"), np.array(predictions))
    np.save(os.path.join(results_path, "ground_truths_train"), np.array(gts))


if __name__ == '__main__':
    infer()
