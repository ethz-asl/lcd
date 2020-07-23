"""
Utility functions for inference.
"""
import os
import numpy as np


def infer_on_test_set(model, test_data_generator, log_dir, epoch):
    """
    Perform inference of the model on the specified test dataset.
    The results will be saved in the log directory and can be viewed using visualize_lines.py.
    :param model: The clustering model.
    :param test_data_generator: The data generator for the test set.
    :param log_dir: The directory where the output will be saved to.
    :param epoch: The current epoch.
    """
    predictions = []
    labels = []
    geometries = []
    bgs = []
    valids = []

    for i in range(test_data_generator.frame_count):
        # Get the data from the generator.
        data, gt = test_data_generator.__getitem__(i)

        # Predict using the neural network.
        output = model.predict(data)

        # Save the results.
        predictions.append(output)
        labels.append(data['labels'])
        geometries.append(data['lines'])
        bgs.append(data['background_mask'])
        valids.append(data['valid_input_mask'])

    # Write the results to numpy array files.
    results_path = os.path.join(log_dir, "results_{:02d}".format(epoch))
    os.mkdir(results_path)
    np.save(os.path.join(results_path, "predictions"), np.array(predictions))
    np.save(os.path.join(results_path, "labels"), np.array(labels))
    np.save(os.path.join(results_path, "geometries"), np.array(geometries))
    np.save(os.path.join(results_path, "backgrounds"), np.array(bgs))
    np.save(os.path.join(results_path, "valids"), np.array(valids))
