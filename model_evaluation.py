# model_evaluation.py
# Description: This file is dedicated to evaluating various machine learning and transfer learning models.
# It contains a suite of functions to compute the accuracy of different models under supervised and unsupervised TL.
# These functions cover a range of models from simple classifiers to more complex transfer learning approaches.
# The module utilizes the 'adapt' library for advanced transfer learning techniques and also includes
# custom implementations such as pnn's. The goal is to assess model performance in different
# transfer learning settings, such as feature-based, instance-based, and parameter-based transfer.

from adapt.feature_based import CORAL, SA, FA
from adapt.instance_based import TrAdaBoost, KLIEP
from adapt.parameter_based import TransferForestClassifier

from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn import svm

from PNN import ProgressiveNeuralNetwork

import time
import numpy as np

# Functions to compute accuracies for various transfer learning models
# Each function takes training data (source and target), the number of trees (or neurons) where applicable,
# and optional domain information, and returns the accuracy score along with the computation time.


def compute_accuracy_relab(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    source_model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0)
    source_model.fit(Xs, ys)

    target_model = TransferForestClassifier(source_model, random_state=0, algo="relab")

    transfer_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    transfer_end_time = time.time()

    tl_score = target_model.predict(Xt)

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, tl_score),\
           transfer_end_time - transfer_start_time


def compute_accuracy_strut(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    source_model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0)
    source_model.fit(Xs, ys)

    target_model = TransferForestClassifier(source_model, random_state=0, algo="strut")

    transfer_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    transfer_end_time = time.time()

    tl_score = target_model.predict(Xt)

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, tl_score),\
           transfer_end_time - transfer_start_time


def compute_accuracy_ser(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    source_model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0)
    source_model.fit(Xs, ys)

    target_model = TransferForestClassifier(source_model, random_state=0, algo="ser")

    transfer_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    transfer_end_time = time.time()

    tl_score = target_model.predict(Xt)

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, tl_score),\
           transfer_end_time - transfer_start_time


def compute_accuracy_tab_rf(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_estimators, domains=None):
    target_model = TrAdaBoost(RandomForestClassifier(n_estimators=nb_estimators, n_jobs=-1, random_state=0), n_estimators=10,
                          Xt=X_transfer, yt=y_transfer, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_tab_linear(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_estimators=None, domains=None):
    target_model = TrAdaBoost(RidgeClassifier(), n_estimators=10,
                          Xt=X_transfer, yt=y_transfer, random_state=0)

    # Fit the TrAdaBoost model to the source data
    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_tab_mlp(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):

    target_model = TrAdaBoost(MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=800, solver="lbfgs"), n_estimators=10,
                          Xt=X_transfer, yt=y_transfer, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_sa_rf(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = SA(RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_sa_linear(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees=None, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = SA(RidgeClassifier(), Xt=X_transfer, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_sa_mlp(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = SA(MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_coral_rf(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = CORAL(RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_coral_linear(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees=None, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = CORAL(RidgeClassifier(), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_coral_mlp(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = CORAL(MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_kliep_rf(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = KLIEP(RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_kliep_linear(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees=None, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = KLIEP(RidgeClassifier(), Xt=x, random_state=0)

    # Fit the Coral model to the source data
    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_kliep_mlp(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    x = np.concatenate((Xt, X_transfer))
    y = np.concatenate((yt, y_transfer))

    target_model = KLIEP(MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000), Xt=x, random_state=0)

    transfer_start_time = time.time()
    target_model.fit(Xs, ys)
    transfer_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           transfer_end_time - transfer_start_time


def compute_accuracy_rf_srconly(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    source_model = RandomForestClassifier(n_estimators=2, n_jobs=-1, random_state=0)

    source_start_time = time.time()
    source_model.fit(Xs, ys)
    source_end_time = time.time()

    print("accuracy:", accuracy_score(yt, source_model.predict(Xt)))

    return accuracy_score(yt, source_model.predict(Xt)), \
           source_end_time - source_start_time


def compute_accuracy_linear_srconly(Xs, ys, X_transfer, y_transfer, Xt, yt, param=None, domains=None):
    source_model = RidgeClassifier()

    source_start_time = time.time()
    source_model.fit(Xs, ys)
    source_end_time = time.time()

    print("accuracy:", accuracy_score(yt, source_model.predict(Xt)))

    return accuracy_score(yt, source_model.predict(Xt)), \
           source_end_time - source_start_time, \



def compute_accuracy_mlp_srconly(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    source_model = MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000)

    source_start_time = time.time()
    source_model.fit(Xs, ys)
    source_end_time = time.time()

    print("accuracy:", accuracy_score(yt, source_model.predict(Xt)))

    return accuracy_score(yt, source_model.predict(Xt)), \
           source_end_time - source_start_time


def compute_accuracy_rf_tgtonly(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    target_model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0)

    target_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time



def compute_accuracy_linear_tgtonly(Xs, ys, X_transfer, y_transfer, Xt, yt, param=None, domains=None, incr_Xs=None, incr_Xt=None):
    target_model = RidgeClassifier()

    target_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_mlp_tgtonly(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    target_model = MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000)

    target_start_time = time.time()
    target_model.fit(X_transfer, y_transfer)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time



def compute_accuracy_rf_all(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    target_model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0)

    target_start_time = time.time()
    target_model.fit(np.concatenate((Xs, X_transfer)), np.concatenate((ys, y_transfer)))
    target_end_time = time.time()


    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_linear_all(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    target_model = RidgeClassifier()

    target_start_time = time.time()
    target_model.fit(np.concatenate((Xs, X_transfer)), np.concatenate((ys, y_transfer)))
    target_end_time = time.time()


    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_mlp_all(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    target_model = MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=2000)

    target_start_time = time.time()
    target_model.fit(np.concatenate((Xs, X_transfer)), np.concatenate((ys, y_transfer)))
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time



def compute_accuracy_pnn(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    target_model = ProgressiveNeuralNetwork()
    for subject in range(len(Xs)):
        target_model.train_new_task(Xs[subject], ys[subject], hidden_dim=nb_neurons, epochs=400)

    target_start_time = time.time()
    target_model.train_new_task(X_transfer, y_transfer, hidden_dim=nb_neurons, epochs=1000)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_fa_rf(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    target_model = FA(RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, random_state=0), Xt=X_transfer, yt=y_transfer, random_state=0)
    target_start_time = time.time()
    target_model.fit(Xs, ys, domains=domains)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_fa_linear(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees=None, domains=None):
    target_model = FA(RidgeClassifier(), Xt=X_transfer, yt=y_transfer, random_state=0)

    target_start_time = time.time()
    target_model.fit(Xs, ys, domains=domains)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_fa_mlp(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    target_model = FA(MLPClassifier(hidden_layer_sizes=(nb_neurons,), random_state=0, max_iter=1000), Xt=X_transfer, yt=y_transfer, random_state=0)

    target_start_time = time.time()
    target_model.fit(Xs, ys, domains=domains)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_fa_svm(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_neurons, domains=None):
    target_model = FA(svm.SVC(), Xt=X_transfer, yt=y_transfer, random_state=0)

    target_start_time = time.time()
    target_model.fit(Xs, ys, domains=domains)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time


def compute_accuracy_fa_hgb(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees, domains=None):
    target_model = FA(HistGradientBoostingClassifier(max_iter=nb_trees), Xt=X_transfer, yt=y_transfer, random_state=0)

    target_start_time = time.time()
    target_model.fit(Xs, ys, domains=domains)
    target_end_time = time.time()

    print("accuracy:", accuracy_score(yt, target_model.predict(Xt)))

    return accuracy_score(yt, target_model.predict(Xt)), \
           target_end_time - target_start_time
