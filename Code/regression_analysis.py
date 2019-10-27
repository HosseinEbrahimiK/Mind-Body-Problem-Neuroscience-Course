import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model


if __name__ == "__main__":
    data_set = pd.read_csv("avg_data.csv")
    data_set2 = pd.read_csv("data.csv")
    corr = data_set2.corr()
    print(corr['correctness'])
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()

    X = np.array(data_set["meanPain"])
    Y = np.array(data_set["correctness"])
    X_p = np.c_[np.ones(np.shape(X)[0]), X]
    A = np.matmul(X_p.T, X_p)
    b = np.matmul(X_p.T, Y)
    w = np.matmul(np.linalg.inv(A), b)

    f = lambda x: w[0] + w[1] * x

    plt.scatter(X, Y, color='black')
    plt.plot(X, f(X), color="blue", linewidth=2)
    plt.xlabel("Average pain rate")
    plt.ylabel("Number of correct response")
    plt.legend(["Regression", "Samples"])
    plt.show()

    print(data_set)


