from metric import *
import numpy as np

class KMeans:
    def __init__(self, data, metric: MetricType, k=3):
        self.k = k
        self.data = data
        self.metric = Metric(metric)
        self.centroids = []
        self.labels = []
        self.lc = []

    def calculate_error(self):
        error = 0.0
        for i, point in enumerate(self.data):
            error += self.metric(point, self.centroids[self.labels[i]]) ** 2
        return error


    def elbow(self, max_k):
        for i in range(1, max_k):
            self.k = i
            self.fit()
            self.lc.append((i, self.calculate_error()))

    def fit(self, max_iter=100):
        history = []
        random_indices = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = [self.data[i] for i in random_indices]
        for _ in range(max_iter):
            labels = []
            for point in self.data:
                distances = [self.metric(point, centroid) for centroid in self.centroids]
                labels.append(np.argmin(distances))
            labels = np.array(labels)
            new_centroids = []
            for i in range(self.k):
                cluster_points = self.data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    rand = np.random.randint(0, len(self.data))
                    new_centroids.append(self.data[rand])
                    labels[rand] = i

            new_centroids = np.array(new_centroids)
            self.labels = np.array(labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

            history.append((enumerate(self.centroids), self.labels))