# utils.py
# Description: This file contains utility functions that support various aspects of the project.
# It includes functions for data manipulation such as normalization, shuffling, sampling, and separating data.
# Additionally, it provides functionalities for loading datasets.
# The module also contains helper functions for saving results and handling transfer learning parameters.

import umap
import pickle
from sklearn.preprocessing import normalize
import numpy as np
import copy
from matplotlib import pyplot as plt
import pandas
import datetime

from constants import *

# Function to learn UMAP embedding space from given data
def learn_space_umap(X, Y, min_dist=0.1, n_neighbors=30):
    mapper = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors).fit(X, Y)
    return mapper


# Function to shuffle data within subjects and sessions
def shuffle_data(X, Y):
    _X = copy.deepcopy(X)
    _Y = copy.deepcopy(Y)
    s_x = []
    s_y = []

    for subject_idx in range(len(_X)):
        s_x.append([])
        s_y.append([])
        for session_idx in range(len(_X[subject_idx])):
            x_np = np.array(_X[subject_idx][session_idx])
            y_np = np.array(_Y[subject_idx][session_idx])

            shuffled_idxes = np.random.permutation(len(_X[subject_idx][session_idx]))
            s_x[-1].append(x_np[shuffled_idxes].tolist())
            s_y[-1].append(y_np[shuffled_idxes].tolist())

    return s_x, s_y


# Function to sample data with replacement within subjects and sessions
def sample_w_rep(X, Y):
    _X = copy.deepcopy(X)
    _Y = copy.deepcopy(Y)
    s_x = []
    s_y = []

    for subject_idx in range(len(_X)):
        s_x.append([])
        s_y.append([])
        for session_idx in range(len(_X[subject_idx])):
            x_np = np.array(_X[subject_idx][session_idx])
            y_np = np.array(_Y[subject_idx][session_idx])

            sampled_idxes = np.random.randint(0, len(_X[subject_idx][session_idx]),
                                              len(_X[subject_idx][session_idx]))
            s_x[-1].append(x_np[sampled_idxes].tolist())
            s_y[-1].append(y_np[sampled_idxes].tolist())

    return s_x, s_y


# Function to apply L2 normalization to each session in each subject
def l2_normalize(X):
    # X is a list of lists (subjects) of lists (sessions) of samples

    # Apply L2 normalization to each session in each subject
    X_normalized = [[normalize(session, norm='l2') for session in subject] for subject in X]

    return X_normalized


# Function to separate dataset into right and left hand data
def get_separate_hands_data(X, Y):
    nb_subjects = len(X)

    X_R = []
    X_L = []
    y_r = []
    y_l = []

    for subject in range(nb_subjects):
        X_R.append([])
        X_L.append([])
        y_r.append([])
        y_l.append([])
        for session in range(len(X[subject])):
            if session % 2 == 0:
                X_R[subject].append(X[subject][session])
                y_r[subject].append(Y[subject][session])
            else:
                X_L[subject].append(X[subject][session])
                y_l[subject].append(Y[subject][session])

    return copy.deepcopy(X_R), copy.deepcopy(X_L), copy.deepcopy(y_r), copy.deepcopy(y_l)


# Function to incrementally add data based on a given percentage
def get_incr_data(X, Y, incr_percentage):
    x = copy.deepcopy(X)
    y = copy.deepcopy(Y)

    nb_subjects = len(y)
    signs = [2, 7, 19, 23]

    new_x = [[None for j in range(len(x[i]))] for i in range(nb_subjects)]
    new_y = [[None for j in range(len(y[i]))] for i in range(nb_subjects)]

    for subject_idx in range(len(y)):
        for session_idx in range(len(y[subject_idx])):
            x_session = np.array(x[subject_idx][session_idx])
            y_session = np.array(y[subject_idx][session_idx]).astype('int16')
            x_session_data = []
            y_session_data = []
            for sign in signs:
                indexes = y_session == sign
                x_session_sign_data = x_session[indexes]
                y_session_sign_data = y_session[indexes]

                percentage_data_count = (len(y_session_sign_data) * incr_percentage) // 100
                x_session_data.append(x_session_sign_data[:percentage_data_count])
                y_session_data.append(y_session_sign_data[:percentage_data_count])

            new_x[subject_idx][session_idx] = np.concatenate(x_session_data).tolist()
            new_y[subject_idx][session_idx] = np.concatenate(y_session_data).tolist()

    return new_x, new_y


# Functions to load different datasets
def load_old_dataset_both():
    with open('dataset/X_Y_both.pkl', 'rb') as f:
        X, Y = pickle.load(f)

    X, Y = shuffle_data(X, Y)

    return get_separate_hands_data(X, Y)


def load_old_dataset_r():
    with open('dataset/X_Y_r.pkl', 'rb') as f:
        X, Y = pickle.load(f)

    X, Y = shuffle_data(X, Y)

    return X, Y


def load_dataset_r():
    X = np.load('dataset/TDF_signs_all.npy', allow_pickle=True)
    Y = np.load('dataset/Y_signs_all.npy', allow_pickle=True)
    X = X.T[RIGHT_HAND]
    Y = Y.T[RIGHT_HAND]
    X = X.tolist()
    Y = Y.tolist()

    for i in range(len(Y)):
        for j in range(len(Y[i])):
            np_ar = np.array(Y[i][j])
            np_ar[np_ar == 2] = 19  # order important !
            np_ar[np_ar == 0] = 2
            np_ar[np_ar == 1] = 7
            np_ar[np_ar == 3] = 23
            X[i][j] = X[i][j].tolist()
            Y[i][j] = [str(e) for e in np_ar.tolist()]

        Y[i] = Y[i].tolist()

    X, Y = shuffle_data(X, Y)
    return X, Y


def load_dataset_l():
    X = np.load('dataset/TDF_signs_all.npy', allow_pickle=True)
    Y = np.load('dataset/Y_signs_all.npy', allow_pickle=True)
    X = X.T[LEFT_HAND]
    Y = Y.T[LEFT_HAND]
    X = X.tolist()
    Y = Y.tolist()

    for i in range(len(Y)):
        for j in range(len(Y[i])):
            np_ar = np.array(Y[i][j])
            np_ar[np_ar == 2] = 19  # order important !
            np_ar[np_ar == 0] = 2
            np_ar[np_ar == 1] = 7
            np_ar[np_ar == 3] = 23
            X[i][j] = X[i][j].tolist()
            Y[i][j] = [str(e) for e in np_ar.tolist()]

        Y[i] = Y[i].tolist()

    X, Y = shuffle_data(X, Y)
    return X, Y


def load_dataset_both():
    Xr, Yr = load_dataset_r()
    Xl, Yl = load_dataset_l()

    return (Xr, Xl), (Yr, Yl)


# Function to extract and structure transfer learning test parameters
def get_transfer_learning_tests_parameters(X, model,
                                           accuracy_per_params_tl,
                                           accuracy_per_params_tl_mean,
                                           compute_time_per_params_tl,
                                           compute_time_per_params_tl_mean,
                                           fig_name, params, subject, transfer_type):
    result = {}
    result['X'] = X
    result['model'] = model
    result['accuracy_per_params_tl'] = accuracy_per_params_tl
    result['accuracy_per_params_tl_mean'] = accuracy_per_params_tl_mean
    result['compute_time_per_params_tl'] = compute_time_per_params_tl
    result['compute_time_per_params_tl_mean'] = compute_time_per_params_tl_mean
    result['fig_name'] = fig_name
    result['params'] = params
    result['subject'] = subject
    result['transfer_type'] = transfer_type
    return result


# Function to save comparison evaluation results of different algorithms and increments
def save_increment_comparison_results(algo1_results, algo2_results, increments):
    sub_name = ""
    first_param = 0  # considering only 1 param
    accuracy_per_params_source_mean = [get_transfer_learning_tests_parameters(
        *algo2_results[i])["accuracy_per_params_tl_mean"][first_param] for i in range(len(algo2_results))]
    accuracy_per_params_tl_mean = [get_transfer_learning_tests_parameters(
        *algo1_results[i])["accuracy_per_params_tl_mean"][first_param] for i in range(len(algo1_results))]
    fig_name = get_transfer_learning_tests_parameters(*algo1_results[0])["fig_name"]
    c_model = get_transfer_learning_tests_parameters(*algo2_results[0])["model"]
    target_model = get_transfer_learning_tests_parameters(*algo1_results[0])["model"]

    date = str(datetime.datetime.now()).replace(".", "").replace("-", "").replace(":", "")

    parameter = "accuracy"

    df = pandas.DataFrame({"increment %": increments, c_model + parameter: accuracy_per_params_source_mean,
                           target_model + " " + parameter: accuracy_per_params_tl_mean})
    df.to_csv(target_model + "_" + parameter + date + sub_name + "_" + fig_name + ".csv", index=False)
    df.plot(x="increment %", marker='o')
    # plt.show()
    plt.savefig(target_model + "_" + parameter + date + sub_name + "_" + fig_name)


# Function to save results of model evaluation
def save_results(X, model, accuracy_per_params_tl,
                 accuracy_per_params_tl_mean, compute_time_per_params_tl,
                 compute_time_per_params_tl_mean,
                 fig_name, params, subject, transfer_type):
    sub_name = ""
    if transfer_type == TRANSFER_INTER_SUBJECT or \
            transfer_type == TRANSFER_SUBJECT_SUBJECT or \
            transfer_type == TRANSFER_INTER_SUBJECT_METRIC or \
            transfer_type == TRANSFER_INTER_SUBJECT_PNN:
        target = "subject"
        target_range = list(range(1, len(X[NON_BOOTSTRAP_INDEX]) + 1))

    elif transfer_type == TRANSFER_INTER_SUBJECT_BOTH or \
            transfer_type == TRANSFER_INTER_HANDS_SUBJECT or \
            transfer_type == TRANSFER_INTER_HANDS_INTER_SUBJECT:
        target = "subject"
        target_range = list(range(1, len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND]) + 1))

    elif transfer_type == TRANSFER_SESSION_INTER_SUBJECT or \
            transfer_type == TRANSFER_SESSION_SUBJECT:
        target = "session"
        target_range = [1]
        sub_name = "_subject" + "_" + str(subject + 1)
    else:
        raise Exception

    date = str(datetime.datetime.now()).replace(".", "").replace("-", "").replace(":", "")

    parameter = "accuracy"
    df = pandas.DataFrame({"params": params, model + " " + parameter: accuracy_per_params_tl_mean})

    df.to_csv(model + "_" + parameter + date + sub_name + "_" + fig_name + ".csv", index=False)
    df.plot(x="params", marker='o')
    plt.savefig(model + "_" + parameter + date + sub_name + "_" + fig_name)

    for param in range(len(params)):
        parameter = "accuracy"
        df = pandas.DataFrame({target: target_range, model + " " + parameter: accuracy_per_params_tl[param]})
        df.to_csv(
            model + "_" + parameter + date + sub_name + "_param_" + str(params[param]) + "_" + fig_name + ".csv",
            index=False)
        df.plot(x=target, marker='o')
        plt.savefig(model + "_" + parameter + date + sub_name + "_param_" + str(params[param]) + "_" + fig_name)
