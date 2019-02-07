"""
The following module contains functions to cluster lines given their embeddings.
"""
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering


def cluster_lines_aggr_clustering(embeddings, num_clusters):
    """ Clusters the num_lines input lines into num_clusters clusters using
        Agglomerative Clustering. Returns an integer label between 0 and
        num_clusters - 1 for each input line, giving the same label to lines
        that are assigned to the same cluster.

    Args:
        embeddings (numpy array of shape (num_lines, embeddings_length)): Vector
            containing the embeddings associated to each line.
        num_clusters (int): Number of clusters to return.

    Returns:
        cluster_labels (numpy array of shape (num_lines, )): Vector containing
            for each line the integer label associated to the cluster to which
            the line was assigned. The labels assume all the integer values
            between 0 and num_cluster - 1. If num_cluster > num_lines an
            exception is raised.
    """
    num_lines = embeddings.shape[0]
    if (num_clusters > num_lines):
        raise ValueError(
            "Trying to extract {0} clusters from {1} lines.".format(
                num_clusters, num_lines))
    aggr_clustering = AgglomerativeClustering(
        n_clusters=num_clusters, affinity='euclidean',
        linkage='ward').fit(embeddings)
    cluster_labels = aggr_clustering.labels_

    return cluster_labels.reshape(num_lines,)


def cluster_lines_kmeans(embeddings, num_clusters):
    """ Clusters the num_lines input lines into num_clusters clusters using
        KMeans. Returns an integer label between 0 and num_clusters - 1 for each
        input line, giving the same label to lines that are assigned to the same
        cluster.

    Args:
        embeddings (numpy array of shape (num_lines, embeddings_length)): Vector
            containing the embeddings associated to each line.
        num_clusters (int): Number of clusters to return.

    Returns:
        cluster_labels (numpy array of shape (num_lines, )): Vector containing
            for each line the integer label associated to the cluster to which
            the line was assigned. The labels assume all the integer values
            between 0 and num_cluster - 1. If num_cluster > num_lines an
            exception is raised.
    """
    num_lines = embeddings.shape[0]
    if (num_clusters > num_lines):
        raise ValueError(
            "Trying to extract {0} clusters from {1} lines.".format(
                num_clusters, num_lines))
    kmeans = KMeans(num_clusters, init='k-means++').fit(embeddings)
    cluster_labels = kmeans.labels_

    return cluster_labels.reshape(num_lines,)


def cluster_lines_affinity_propagation(embeddings):
    """ Clusters the num_lines input lines into clusters using
        AffinityPropagation. Returns an integer label between 0 and
        num_clusters - 1 for each input line, where num_clusters is the number
        of clusters found. Lines that are assigned to the same cluster are given
        the same label.

    Args:
        embeddings (numpy array of shape (num_lines, embeddings_length)): Vector
            containing the embeddings associated to each line.

    Returns:
        cluster_labels (numpy array of shape (num_lines, )): Vector containing
            for each line the integer label associated to the cluster to which
            the line was assigned. The labels assume all the integer values
            between 0 and num_cluster - 1.
        num_clusters (int): Number of clusters found.
    """
    num_lines = embeddings.shape[0]
    affinity_propagation = AffinityPropagation().fit(embeddings)
    cluster_labels = affinity_propagation.labels_
    num_clusters = affinity_propagation.cluster_centers_indices_.shape[0]

    return cluster_labels.reshape(num_lines,), num_clusters
