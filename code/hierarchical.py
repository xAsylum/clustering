from enum import Enum
from metric import *
import numpy as np


class LinkageType(Enum):
    Singular = 1
    Full = 2
    Mean = 3
    Centroid = 4
    Ward = 5

class HierarchicalClusterization:
    def __init__(self, linkage_type: LinkageType, metric_type: MetricType, data, k):
        self.data = data
        self.linkage_type = linkage_type
        self.metric = Metric(metric_type)
        self.k = k

    def compute_distance_matrix(self):
        n = len(self.data)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.metric(self.data[i], self.data[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def fit(self):
        n_clusters = self.k
        n_samples = len(self.data)

        dist_matrix = self.compute_distance_matrix()

        active_clusters = list(range(n_samples))
        cluster_sizes = {i: 1 for i in range(n_samples)}
        cluster_assignments = {i: [i] for i in range(n_samples)}

        history = []

        next_cluster_id = n_samples

        while len(active_clusters) > n_clusters:
            min_dist = float('inf')
            min_i, min_j = -1, -1

            for i_idx, i in enumerate(active_clusters[:-1]):
                for j in active_clusters[i_idx + 1:]:
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        min_i, min_j = i, j

            new_cluster_id = next_cluster_id
            next_cluster_id += 1

            cluster_assignments[new_cluster_id] = cluster_assignments[min_i] + cluster_assignments[min_j]
            cluster_sizes[new_cluster_id] = cluster_sizes[min_i] + cluster_sizes[min_j]

            history.append((min_i, min_j, min_dist, cluster_sizes[new_cluster_id]))
            for k in active_clusters:
                if k != min_i and k != min_j:
                    new_dist = self.lance_williams_update(dist_matrix, min_i, min_j, k, cluster_sizes)

                    if new_cluster_id >= dist_matrix.shape[0]:
                        dist_matrix = np.pad(
                            dist_matrix,
                            ((0, new_cluster_id - dist_matrix.shape[0] + 1),
                             (0, new_cluster_id - dist_matrix.shape[0] + 1)),
                            'constant',
                            constant_values=0
                        )

                    dist_matrix[new_cluster_id, k] = new_dist
                    dist_matrix[k, new_cluster_id] = new_dist

            active_clusters.remove(min_i)
            active_clusters.remove(min_j)
            active_clusters.append(new_cluster_id)

        labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster_id in enumerate(active_clusters):
            for sample_idx in cluster_assignments[cluster_id]:
                if sample_idx < n_samples:
                    labels[sample_idx] = cluster_idx

        return labels, history

    def lance_williams_update(self, dist_matrix, i, j, k, cluster_sizes):
        n_i = cluster_sizes[i]
        n_j = cluster_sizes[j]
        n_k = cluster_sizes[k]

        if self.linkage_type == LinkageType.Singular:
            # alpha_i = alpha_j = 0.5, beta = 0, gamma = -0.5
            return min(dist_matrix[i, k], dist_matrix[j, k])

        elif self.linkage_type == LinkageType.Full:
            # alpha_i = alpha_j = 0.5, beta = 0, gamma = 0.5
            return max(dist_matrix[i, k], dist_matrix[j, k])

        elif self.linkage_type == LinkageType.Mean:
            # alpha_i = n_i/(n_i+n_j), alpha_j = n_j/(n_i+n_j), beta = 0, gamma = 0
            return (n_i * dist_matrix[i, k]
                    + n_j * dist_matrix[j, k]) / (n_i + n_j)

        elif self.linkage_type == LinkageType.Centroid:
            # alpha_i = n_i/(n_i+n_j), alpha_j = n_j/(n_i+n_j), beta = -(n_i*n_j)/((n_i+n_j)^2), gamma = 0
            alpha_i = n_i / (n_i + n_j)
            alpha_j = n_j / (n_i + n_j)
            beta = -(n_i * n_j) / ((n_i + n_j) ** 2)
            return alpha_i * dist_matrix[i, k] + alpha_j * dist_matrix[j, k] + beta * dist_matrix[i, j]

        elif self.linkage_type == LinkageType.Ward:
            # alpha_i = (n_i+n_k)/(n_i+n_j+n_k), alpha_j = (n_j+n_k)/(n_i+n_j+n_k),
            # beta = -n_k/(n_i+n_j+n_k), gamma = 0
            alpha_i = (n_i + n_k) / (n_i + n_j + n_k)
            alpha_j = (n_j + n_k) / (n_i + n_j + n_k)
            beta = -n_k / (n_i + n_j + n_k)
            return alpha_i * dist_matrix[i, k] + alpha_j * dist_matrix[j, k] + beta * dist_matrix[i, j]

