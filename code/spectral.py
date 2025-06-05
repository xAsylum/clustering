import numpy as np
from numpy.ma import product

from hierarchical import *
from kmeans import KMeans

from metric import *

def eigen(M):
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvectors, eigenvalues

class SpectralClusterization:
    def __init__(self, metric_type: MetricType, data, k, s):
        self.labels = None
        self.raw_data = data
        self.data = data
        self.metric = Metric(metric_type)
        self.metric_type = metric_type
        self.sigma = s
        self.k = k

    def affinity_matrix(self):
        n = len(self.data)
        aff_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                aff_matrix[i, j] = np.exp(-(self.sigma * self.metric(self.data[i], self.data[j]) ** 2))

        return aff_matrix

    def laplacian(self):
        n = len(self.data)
        A = self.affinity_matrix()
        D = np.zeros((n, n))
        for i in range(n):
            D[i, i] = 1 / np.sqrt(np.sum(A[i, :]))

        return np.dot(np.dot(D, A), D)

    def fit(self):
        L = self.laplacian()
        eigenvectors, eigenvalues = eigen(L)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]

        spectral_embeddings = eigenvectors[:, :self.k]

        #model = KMeans(spectral_embeddings, self.metric_type, self.k)
        #model.fit()
        model = HierarchicalClusterization(LinkageType.Full, self.metric_type, spectral_embeddings, self.k)
        labels, history = model.fit()
        self.labels = labels