"""
Callback functions for training.
"""
import tensorflow.keras as tf_keras
import inference
import model


class LayerUnfreezeCallback(tf_keras.callbacks.Callback):
    def __init__(self, loss, opt, metrics):
        super().__init__()

        self.loss = loss
        self.opt = opt
        self.metrics = metrics

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= 7:
            transfer_layers = ["block3_conv1", "block3_conv2", "block3_conv3"]
            for layer_name in transfer_layers:
                self.model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = \
                    True
                print("Unfreezing layer {}.".format(layer_name))

        if epoch >= 13:
            transfer_layers = ["block2_conv1", "block2_conv2"]
            for layer_name in transfer_layers:
                self.model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = \
                    True
                print("Unfreezing layer {}.".format(layer_name))

        if epoch >= 19:
            transfer_layers = ["block1_conv1", "block1_conv2"]
            for layer_name in transfer_layers:
                self.model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = \
                    True
                print("Unfreezing layer {}.".format(layer_name))

        self.model.compile(loss=self.loss,
                           optimizer=self.opt,
                           metrics=self.metrics,
                           experimental_run_tf_function=False)


class LearningRateCallback(tf_keras.callbacks.Callback):
    def __init__(self, decay):
        super().__init__()
        self.decay = decay

    def on_epoch_begin(self, epoch, logs=None):
        for key in sorted(self.decay.keys()):
            if epoch >= key:
                print("Setting learning rate to {}.".format(self.decay[key]))
                self.model.optimizer.learning_rate = self.decay[key]

        if epoch + 1 in self.decay:
            print("Setting learning rate to {}.".format(self.decay[epoch + 1]))
            self.model.optimizer.learning_rate = self.decay[epoch + 1]


class InferenceCallback(tf_keras.callbacks.Callback):
    def __init__(self, test_data_generator, log_dir):
        super().__init__()

        self.test_data_generator = test_data_generator
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        print("Inference on test set.")
        inference.infer_on_test_set(self.model, self.test_data_generator, self.log_dir, epoch + 1)
        print("Inference done.")


class SaveImageWeightsCallback(tf_keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.model.get_layer("image_features").save_weights(self.path)


class SaveCallback(tf_keras.callbacks.Callback):
    def __init__(self, path, cluster=False):
        super().__init__()
        self.path = path
        self.cluster = cluster

    def on_epoch_end(self, epoch, logs=None):
        if self.cluster:
            model.save_cluster_model(self.model, self.path.format(epoch + 1))
        else:
            model.save_line_net_model(self.model, self.path.format(epoch + 1))