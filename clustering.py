import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """

    data = df[[features[0], features[1]]].to_numpy()
    sum = data.sum(axis=0)
    min = data.min(axis=0)
    data = (data - min)/sum
    data = add_noise(data)

    return data


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    prev_centroids = choose_initial_centroids(data,k)
    labels = assign_to_clusters(data,prev_centroids)
    current_centroids = recompute_centroids(data, labels, k)


    while not (np.array_equal(prev_centroids, current_centroids)):
        prev_centroids = current_centroids
        labels = assign_to_clusters(data,prev_centroids)
        current_centroids = recompute_centroids(data, labels, k)

    prev_centroids  = np.around(prev_centroids, decimals=3)
    return labels, np.around(current_centroids,3)


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """

    k = centroids.shape[0]
    plt.scatter(data[:, 0], data[:, 1], c=labels, linewidths=0.5)
    plt.xlabel('$cnt$')
    plt.ylabel('$hum$')
    plt.title(F'Results for kmeans with k = {k}')
    for i in range(len(centroids)):
        plt.scatter(centroids[i, 0], centroids[i, 1], color='white', edgecolors='black', marker='*')

    plt.show()
    plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = np.linalg.norm(y-x)
    return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    n = data.shape[0]
    k = centroids.shape[0]
    distances = get_distances(data, n, k, centroids)
    labels = split_into_clusters(distances, n, k)
    return labels

def split_into_clusters(distances, n, k): #checked?
    labels = distances.argmin(axis=0)
    return labels


def get_distances(data, n, k, centroids):
    distance = np.ones(shape=(k, n))
    for index, pair in enumerate(data):
        for i in range(k):
            distance[i, index] = (dist(pair, centroids[i]))
    return distance



def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.zeros(shape=(k, 2))
    counters = np.zeros(shape=(k, 1))
    for i, pair in enumerate(data):
        for j in range(k):
            if labels[i] == j:
                centroids[j, 0] += pair[0]
                centroids[j, 1] += pair[1]
                counters[j, 0] += 1

    centroids /= counters
    return centroids

