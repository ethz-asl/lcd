""" Display statistics about the embeddings obtained for each instance in a
    certain dataset, i.e., it displays statistics about the potential clusters
    that can be formed in the embedding space (e.g., mean and standard deviation
    of the distances from the 'mean embedding', max and min 'intra-instance'
    distance, etc.).
"""

import argparse
import numpy as np
import operator
import sys


def display_statistics(embeddings_file_path,
                       instances_file_path,
                       output_statistics_file_path=None):
    """ Displays statistics about the potential clusters that can be formed in
        the embedding space given the retrieved embeddings and the ground-truth
        instances.

    Args:
        embeddings_file_path (string): Path of the .npy file containing the
            retrieved embeddings.
        instances_file_path (string): Path of the .npy file containing the
            ground-truth instance labels associated to the lines for which
            embeddings were retrieved. NOTE: the order of the ground-truth
            instance labels should match the one of the embeddings, i.e., the
            i-th line of the ground-truth instance labels file should contain
            the ground-truth instance label of the line that is at the i-th line
            of the embeddings file.
        output_statistics_file_path (string): If not None, the statistics are
            printed to the file output_statistics_file. Otherwise, statistics
            are displayed on screen.
    """
    # Load embeddings file.
    try:
        embeddings = np.load(embeddings_file_path)
    except:
        print("Error while opening embeddings file at {}. Exiting.".format(
            embeddings_file_path))
        return
    # Load ground-truth instances file.
    try:
        ground_truth_instances = np.load(instances_file_path)
    except:
        print("Error while opening ground-truth instances file at {}. ".format(
            instances_file_path) + "Exiting.")
        return
    # Check that embeddings and ground-truth labels are compatible.
    assert (embeddings.shape[0] == ground_truth_instances.shape[0])

    # Assign each embedding to its ground-truth instance.
    clusters = dict()
    for i in range(embeddings.shape[0]):
        if ground_truth_instances[i] in clusters:
            clusters[ground_truth_instances[i]].append(embeddings[i])
        else:
            clusters[ground_truth_instances[i]] = []

    # - Retrieve mean and standard deviation, of the distances between the
    #   embeddings of lines and the 'mean embedding' of all the lines with the
    #   same ground-truth instance label.
    # - Also, store the max and min 'intra-instance' distance, i.e., the
    #   max/min distance between the embeddings of any two lines that have the
    #   same ground-truth instance label.
    # - For each instance, store the smallest distance between the embedding of
    #   any of the lines from that instance and the embeddings of any of the
    #   lines from any other instances (i.e., 'extra-instance' distance). Also,
    #   store the instance to which the line that achieves this distance
    #   belongs.
    # - For each instance, store the smallest distance between the
    #   'mean embedding' of that instance and the embeddings of any of the
    #   lines from any other instances (i.e., 'extra-to-(mean embedding)'
    #   distance). Also, store the instance to which the line that achieves this
    #   distance belongs.
    # - For each instance, find the (up to) 5 closest instances in the embedding
    #   space (in terms of 'mean embedding'-to-'mean embedding' distance).
    avg_dist_from_mean = dict()
    std_dist_from_mean = dict()
    mean_embedding = dict()
    max_intra_instance_dist = dict()
    min_intra_instance_dist = dict()
    min_extra_instance_dist = dict()
    min_extra_to_mean_embedding_dist = dict()
    dist_from_other_instances_mean = dict()
    smallest_dist_from_other_instances_mean = dict()

    for instance in clusters.keys():
        # Compute 'mean embedding'.
        mean_embedding[instance] = np.mean(clusters[instance], axis=0)
        # Compute average L2 distance from the mean embedding.
        avg_dist_from_mean[instance] = np.mean(
            np.linalg.norm(
                clusters[instance] - mean_embedding[instance], axis=1))
        # Compute L2 standard deviation of the distances from the mean
        # embedding.
        std_dist_from_mean[instance] = np.std(
            np.linalg.norm(
                clusters[instance] - mean_embedding[instance], axis=1))
        # Compute max and min 'intra-instance' L2 distance.
        for i in range(len(clusters[instance]) - 1):
            for j in range(i + 1, len(clusters[instance])):
                curr_dist = np.linalg.norm(
                    clusters[instance][i] - clusters[instance][j])
                # Max distance.
                if instance in max_intra_instance_dist:
                    if curr_dist > max_intra_instance_dist[instance]:
                        max_intra_instance_dist[instance] = curr_dist
                else:
                    max_intra_instance_dist[instance] = curr_dist
                # Min distance.
                if instance in min_intra_instance_dist:
                    if curr_dist < min_intra_instance_dist[instance]:
                        min_intra_instance_dist[instance] = curr_dist
                else:
                    min_intra_instance_dist[instance] = curr_dist

    for instance in clusters.keys():
        # Compute min 'extra-instance' L2 distance, min
        # extra-to-'mean embedding' L2 distance and
        # 'mean embedding'-to-.'mean embedding' L2 distance.
        if instance not in dist_from_other_instances_mean:
            dist_from_other_instances_mean[instance] = dict()
        for other_instance in clusters.keys():
            if other_instance == instance:
                continue
            # 'Mean embedding'-to-.'mean embedding' distance.
            dist_from_other_instances_mean[
                instance][other_instance] = np.linalg.norm(
                    mean_embedding[instance] - mean_embedding[other_instance])
            for j in range(len(clusters[other_instance])):
                # Extra-to-'mean embedding' distance.
                curr_dist = np.linalg.norm(
                    mean_embedding[instance] - clusters[other_instance][j])
                if instance in min_extra_to_mean_embedding_dist:
                    if (curr_dist <
                            min_extra_to_mean_embedding_dist[instance][0]):
                        min_extra_to_mean_embedding_dist[instance] = (
                            curr_dist, other_instance)
                else:
                    min_extra_to_mean_embedding_dist[instance] = (
                        curr_dist, other_instance)

                for i in range(len(clusters[instance])):
                    # 'Extra-instance' distance.
                    curr_dist = np.linalg.norm(
                        clusters[instance][i] - clusters[other_instance][j])
                    if instance in min_extra_instance_dist:
                        if curr_dist < min_extra_instance_dist[instance][0]:
                            min_extra_instance_dist[instance] = (curr_dist,
                                                                 other_instance)
                    else:
                        min_extra_instance_dist[instance] = (curr_dist,
                                                             other_instance)
        # Sort 'mean embedding'-to-.'mean embedding' distance to only keep
        # the 5 smallest ones.
        smallest_dist_from_other_instances_mean[instance] = sorted(
            dist_from_other_instances_mean[instance].items(),
            key=operator.itemgetter(1))[:5]

    # Print/display statistics.
    if (output_statistics_file_path is not None):
        # Create parent directory if nonexistent (based on
        # https://stackoverflow.com/a/12517490).
        if not os.path.exists(os.path.dirname(output_statistics_file_path)):
            try:
                os.makedirs(os.path.dirname(output_statistics_file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        # Open file.
        f = open(output_statistics_file_path, 'w')
    else:
        f = sys.stdout

    # Write statistics.
    f.write("The {0} embeddings retrieved belong to {1} different instances ".
            format(len(embeddings), len(clusters.keys())) +
            "and have the following distribution:\n")
    for instance_label in clusters.keys():
        f.write("- Instance {0}: {1} occurrences\n".format(
            instance_label, len(clusters[instance_label])))
    for instance_label in clusters.keys():
        f.write("\nInstance {}:\n".format(instance_label))
        f.write("- Average distance from mean embedding: {:.7f}\n".format(
            avg_dist_from_mean[instance_label]))
        f.write(
            "- Standard deviation of distances from mean embedding: {:.7f}\n".
            format(std_dist_from_mean[instance_label]))
        f.write("- Max 'intra-instance' distance: {:.7f}\n".format(
            max_intra_instance_dist[instance_label]))
        f.write("- Min 'intra-instance' distance: {:.7f}\n".format(
            min_intra_instance_dist[instance_label]))
        f.write("- Min 'extra-instance' distance: {:.7f} ".format(
            min_extra_instance_dist[instance_label][0]) +
                "(extra instance is {})\n".format(
                    min_extra_instance_dist[instance_label][1]))
        f.write("- The embedding from another instance closest to the mean "
                "embedding belongs to instance {} ".format(
                    min_extra_to_mean_embedding_dist[instance_label][1]) +
                "and has distance: {:.7f}\n".format(
                    min_extra_to_mean_embedding_dist[instance_label][0]))
        f.write("- The 5 closest instances in terms of "
                "'mean embedding'-to-'mean embedding' distance are:\n")
        for closest_instance in smallest_dist_from_other_instances_mean[
                instance_label]:
            f.write("  * {0}: d({1}, {0}) = {2:.7f}\n".format(
                closest_instance[0], instance_label, closest_instance[1]))

    if (output_statistics_file_path is not None):
        # Close file.
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Displays statistics about the potential clusters that can '
        'be formed with the embeddings retrieved.')
    parser.add_argument(
        "-embeddings_file_path",
        help="Path of the embeddings file.",
        required=True)
    parser.add_argument(
        "-instances_file_path",
        help="Path of the ground-truth instance file.",
        required=True)
    parser.add_argument(
        "-output_file_path",
        help="Path where to save the statistics. If not passed, statistics are "
        "displayed to screen.")

    args = parser.parse_args()
    if (args.embeddings_file_path):
        embeddings_file_path = args.embeddings_file_path
    if (args.instances_file_path):
        instances_file_path = args.instances_file_path
    if (args.output_file_path):
        output_file_path = args.output_file_path
        display_statistics(
            embeddings_file_path=embeddings_file_path,
            instances_file_path=instances_file_path,
            output_statistics_file_path=output_file_path)
    else:
        display_statistics(
            embeddings_file_path=embeddings_file_path,
            instances_file_path=instances_file_path)
