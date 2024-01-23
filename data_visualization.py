# data_visualization.py
# Description: This files provides functions for visualizing the data using different dimensionality reduction techniques
# and plotting the distribution of labels across subjects. It supports visualization with PCA, Kernel PCA, and UMAP,
# and offers the ability to visualize data both per subject and for all subjects collectively.


from matplotlib import pyplot as plt
import pandas
import umap
import seaborn as sns
import numpy as np

from sklearn.decomposition import KernelPCA, PCA


# Visualize data per subject using PCA, Kernel PCA, or UMAP
def visualise_data_per_subject(algorithm_name, kernel_name, X, Y):
    sign_mapping = {"2": 1, "7": 2, "19": 3, "23": 4}
    # Choose the dimensionality reduction algorithm
    if algorithm_name.lower() == 'pca':
        algorithm = PCA(n_components=2)
    elif algorithm_name.lower() == 'kpca':
        algorithm = KernelPCA(n_components=2, kernel=kernel_name)
    elif algorithm_name.lower() == 'umap':
        algorithm = umap.UMAP(min_dist=0.1)
    else:
        raise ValueError('Invalid algorithm name')

    custom_palette = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red'}

    nb_subjects = len(X)

    # Concatenate all the data
    X_all = np.concatenate([np.concatenate(X[subject]) for subject in range(nb_subjects)])
    Y_all = np.concatenate([np.concatenate(Y[subject]) for subject in range(nb_subjects)])
    Y_all = [sign_mapping[e] for e in Y_all]

    # Perform dimensionality reduction
    algorithm.fit(X_all)

    # Create a 3x5 grid of subplots
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))

    for subject in range(nb_subjects):
        X_subject = np.concatenate(X[subject])
        Y_subject = np.concatenate(Y[subject])
        Y_subject = [sign_mapping[e] for e in Y_subject]

        X_transformed = algorithm.transform(X_subject)

        df = pandas.DataFrame(X_transformed, columns=['comp-1', 'comp-2'])
        df['y'] = Y_subject

        # Determine the row and column in the subplot grid
        row = subject // 5
        col = subject % 5

        # Use seaborn for the scatterplot
        sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(), palette=custom_palette, data=df, legend=False,
                        ax=axs[row, col])
        axs[row, col].set_title("Subject " + str(subject + 1))
        axs[row, col].set_xlabel('')  # Remove x-axis label
        axs[row, col].set_ylabel('')  # Remove y-axis label

    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


# Visualize data for all subjects using selected dimensionality reduction technique
def visualise_data(algorithm_name, kernel_name, X, Y):
    sign_mapping = {"2": 1, "7": 2, "19": 3, "23": 4}

    # Choose the dimensionality reduction algorithm
    if algorithm_name.lower() == 'pca':
        algorithm = PCA(n_components=2)
    elif algorithm_name.lower() == 'kpca':
        algorithm = KernelPCA(n_components=2, kernel=kernel_name)
    elif algorithm_name.lower() == 'umap':
        algorithm = umap.UMAP(min_dist=0.1, n_neighbors=15)
    else:
        raise ValueError('Invalid algorithm name')

    custom_palette = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red'}

    nb_subjects = len(X)

    # Concatenate all the data
    X_all = np.concatenate([np.concatenate(X[subject]) for subject in range(nb_subjects)])
    Y_all = np.concatenate([np.concatenate(Y[subject]) for subject in range(nb_subjects)])
    Y_all = [sign_mapping[e] for e in Y_all]

    # Perform dimensionality reduction
    X_transformed = algorithm.fit_transform(X_all)
    # Create a DataFrame for convenience
    df = pandas.DataFrame(X_transformed, columns=['comp-1', 'comp-2'])
    df['y'] = Y_all

    # Use seaborn for the scatterplot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(), palette=custom_palette, data=df, legend=False)
    plt.title("All Subjects")
    plt.show()


# Function to compute and visualize label balance in the dataset
def compute_label_balance(Y):
    nb_subjects = len(Y)
    signs = [2, 7, 19, 23]
    nb_signs = len(signs)  # sign count
    ys = []
    conc_ys = np.concatenate(np.concatenate(Y)).astype('int16')
    for subject_data in Y:
        ys.append(np.concatenate(subject_data).astype('int16'))

    sign_percentages = []
    for subject_data in ys:
        sign_percentages.append([])
        for sign in signs:
            percentage = np.count_nonzero(subject_data == sign) / len(conc_ys)
            sign_percentages[-1].append(percentage)

    x = range(1, nb_subjects + 1)
    y = np.array(sign_percentages).T
    # plt.figure(figsize=(10,6))
    y_offset = np.zeros(nb_subjects)
    for sign_idx in range(nb_signs):
        plt.bar(x, y[sign_idx], label=str(signs[sign_idx]), bottom=y_offset)
        y_offset = y_offset + y[sign_idx]

    plt.xlabel("Subject")
    plt.ylabel("Quantity ratio")
    plt.legend()

    plt.show()

    sign_percentages = []
    for sign in signs:
        percentage = np.count_nonzero(conc_ys == sign) / len(conc_ys)
        sign_percentages.append(percentage)

    plt.figure(figsize=(10, 5))
    plt.xlabel("Sign")
    plt.ylabel("Quantity ratio")
    plt.bar(list(str(sign) for sign in signs), sign_percentages, color=['blue', 'orange', 'green', 'red'])
    plt.show()