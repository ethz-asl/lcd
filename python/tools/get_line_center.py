import numpy as np

def get_line_center(labels_batch):
    """ Convert an input batch of lines with endpoints and instance label to a batch of lines with only center of the line and instance label.
    Args:
        labels_batch: tensor of shape (batch_size, 7), with each row in the
                      format
                        [start point (3x)] [end point (3x)] [instance label]
    Returns:
        pairwise_distances: tensor of shape (batch_size, 4), with each row in
                            the format
                              [center point (3x)] [instance label]
    """
    assert labels_batch.shape[1] == 7, "{}".format(labels_batch.shape)
    # Obtain center (batch_size, 6)
    center_batch = np.hstack(
        labels_batch[:, [[i], [i + 3]]].mean(axis=1) for i in range(3))
    # Concatenate instance labels column
    return np.hstack((center_batch, labels_batch[:, [-1]]))
