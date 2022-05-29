# import os
from csv import reader

import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_style
from sklearn.decomposition import PCA


def save_pca_images(n_components, read_filepath, write_filepath):

    rows, img_ids = get_rows_array(read_filepath)
    rows = get_sample(rows, 10000)

    X = rows
    X = X / 255

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_tilde = pca.transform(rows)

    # image_shape = (210, 160)

    with open(write_filepath, "w") as file:
        for i in range(0, len(X_tilde)):
            # pca_d = X_tilde[i, :].dot(pca.components_)
            file.write(img_ids[i] + ",")
            file.write(",".join(str(x) for x in X_tilde[i]))
            file.write("\n")


def plot_feature_images(X, X_tilde, pca, image_shape):
    set_style("white")

    plt.show()
    fig, ax = plt.subplots(
        1, 7, figsize=(20, 15), subplot_kw={"xticks": (), "yticks": ()}
    )
    ax[0].imshow(X[100].reshape(image_shape), cmap="gray")
    ax[0].set_title("Original Image", fontsize=16)
    i = 1
    for n_components in [10, 50, 100, 500, 1000, 3000]:
        ax[i].imshow(
            X_tilde[100, :n_components]
            .dot(pca.components_[:n_components, :])
            .reshape(image_shape),
            cmap="gray",
        )
        ax[i].set_title(
            str(n_components)
            + " Components:\n"
            + "Cumulative\nExplained\nVariance\nRatio = "
            + str(
                np.round(100 * np.sum(pca.explained_variance_ratio_[:n_components]), 2)
            )
            + "%",
            fontsize=16,
        )

        i = i + 1


def plot_explained_variance(pca):
    # make the explained variance curve
    plt.figure(figsize=(10, 8))

    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
    )

    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=18)
    plt.xlabel("Number of Principal Components", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlim(0, 500)

    plt.show()


def get_sample(rows, sample_n):
    # ran some trials and found times it took to run PCA on various sample sizes
    # sample of 7040    3m 19.4s
    # sample of 10000   6m 20.5s
    # sample of 15000   15m 43.8s
    # sample of 20000   35m 51.1s
    ind = np.random.randint(len(rows), size=sample_n)
    return rows[ind, :]


def get_rows_array(filepath):

    with open(filepath) as obj:
        csv_reader = reader(obj)
        rows_list = []
        img_ids = []
        for row in csv_reader:
            img_ids.append(row[0])
            rr = np.array([int(r) for r in row[1:]])
            rows_list.append(rr)

    rows = np.array(rows_list)
    return rows, img_ids


def perform_PCA(filepath):

    rows, img_ids = get_rows_array(filepath)

    rows = get_sample(rows, sample_n=10000)

    X = rows
    X = X / 255

    pca = PCA()
    pca.fit(X)
    pca.components_.shape

    plot_explained_variance(pca)

    image_shape = (210, 160)
    X_tilde = pca.transform(X)
    # X_tilde[0, :].dot(pca.components_).reshape(image_shape)

    plot_feature_images(X, X_tilde, pca, image_shape)
