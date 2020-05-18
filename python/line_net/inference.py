import os
import numpy as np

from tensorflow.keras.models import load_model

from datagenerator_framewise import LineDataGenerator
from datagenerator_framewise import generate_data
from model import image_pretrain_model
from model import line_net_model_4


def infer():
    test_files = "/nvme/line_ws/test"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (64, 96, 3)
    max_line_count = 150
    bg_classes = [0, 1, 2, 20, 22]

    log_dir = "/home/felix/line_ws/src/line_tools/python/line_net/logs/180520_0018"
    epoch = 1

    model, loss, opt, metrics = line_net_model_4(line_num_attr, max_line_count, img_shape)
    model.get_layer("image_features").load_weights("/home/felix/line_ws/src/line_tools/python/line_net/weights/image_weights.hdf5")

    # transfer_layers = ["block3_conv1", "block3_conv2", "block3_conv3"]
    # for layer_name in transfer_layers:
    #     model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = True
    #     print("Unfreezing layer {}.".format(layer_name))
    model.load_weights(os.path.join(log_dir, "weights.{}.hdf5".format(epoch)), by_name=True, skip_mismatch=False)

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics,
                  experimental_run_tf_function=False)

    model.summary()
    # model.get_layer("image_features").trainable = False
    # for layer in model.get_layer("image_features").layers:
    #     layer.trainable = False
    # model.get_layer("image_features").save_weights("image_weights.hdf5")
    # model.get_layer("image_features").load_weights("image_weights.hdf5")

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
