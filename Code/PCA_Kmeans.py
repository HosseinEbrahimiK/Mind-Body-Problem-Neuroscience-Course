from random import normalvariate
import numpy as np
from numpy.dual import norm
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def random_unit_vector(n):
    un_normalized = [normalvariate(0, 1) for _ in range(n)]
    the_norm = sqrt(sum(x * x for x in un_normalized))
    return [x / the_norm for x in un_normalized]


def first_pc(a):
    epsilon = 1e-10
    a = np.array(a, dtype=float)
    cov = a.dot(a.T)
    n = cov.shape[0]
    x = random_unit_vector(n)
    current_v = x

    while True:
        last_v = current_v
        current_v = np.dot(cov, last_v)
        current_v = current_v / norm(current_v)

        if abs(np.dot(current_v, last_v)) > 1 - epsilon:
            return current_v


if __name__ == "__main__":

    X = pd.read_csv("/Users/joliaserm/Desktop/NeuroScience/report_analyze_project/data.csv")
    X = np.matrix(X)
    numbers = [2]
    MATRIX = 0
    for k in range(len(numbers)):
        v = list()
        v.append(first_pc(X))

        for i in range(1, numbers[k]):
            m = 0
            for j in range(i):
                m += np.matmul(np.matrix(v[j]).T, np.matrix(v[j]))

            mat = np.matmul(m, X)
            mat = X - mat
            v.append(first_pc(mat))

        MATRIX = v[0]
        for i in range(1, numbers[k]):
            MATRIX = np.c_[MATRIX, v[i]]

        x_axis = MATRIX[:, 0]
        y_axis = MATRIX[:, 1]

        plt.scatter(x_axis, y_axis, color='blue')
        plt.xlabel("First Component of PCA")
        plt.ylabel("Second Component of PCA")
        plt.show()

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(MATRIX)

    plt.scatter(MATRIX[:, 0], MATRIX[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.xlabel("First Component of PCA")
    plt.ylabel("Second Component of PCA")
    plt.show()
