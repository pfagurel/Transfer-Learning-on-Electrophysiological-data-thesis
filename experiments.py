# experiments.py
# Description: This file includes functions to conduct various transfer learning experiments, leveraging models
# and utilities defined in other modules. It supports evaluating model performance across different transfer learning
# scenarios, such as inter-subject, intra-subject, and incremental learning. It uses configurations from
# `model_parameters.py` to iterate over different model parameters and employs functions from `utils.py` for data
# manipulation.

from utils import *
from model_parameters import *
import pandas


# Compute accuracy for inter-subject transfer learning
def compute_accuracy_inter_subject(X, Y, param, model, nb_sessions=1, session=0):
    if nb_sessions < 1 or nb_sessions > 5:
        raise Exception("exceeded range")

    nb_subjects = len(X[NON_BOOTSTRAP_INDEX])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs, ys, Xt, yt, X_transfer, y_transfer = None, None, None, None, None, None

        Xs = copy.deepcopy(X[NON_BOOTSTRAP_INDEX])
        ys = copy.deepcopy(Y[NON_BOOTSTRAP_INDEX])
        Xt = Xs.pop(subject)
        yt = ys.pop(subject)
        if nb_sessions > 1:
            X_transfer = copy.deepcopy(Xt[:nb_sessions])
            y_transfer = copy.deepcopy(yt[:nb_sessions])
        else:
            X_transfer = Xt.pop(session % len(Xt))
            y_transfer = yt.pop(session % len(yt))

        domains = np.array([i for i in range(len(Xs)) for sess in Xs[i] for sample in sess])

        Xs = np.array([sample for subject in Xs for sess in subject for sample in sess])
        ys = np.array([sample for subject in ys for sess in subject for sample in sess]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')

        if nb_sessions > 1:
            X_transfer = np.concatenate(X_transfer)
            y_transfer = np.concatenate(y_transfer).astype('int16')
        else:
            X_transfer = np.array(X_transfer)
            y_transfer = np.array(y_transfer).astype('int16')

        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Compute accuracy considering both right and left hand data
def compute_accuracy_inter_subject_both(X, Y, param, model, nb_sessions=1, session=0):
    if nb_sessions < 1 or nb_sessions > 5:
        raise Exception("exceeded range")

    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs, ys, Xt, yt, X_transfer, y_transfer = None, None, None, None, None, None

        temp_xr = copy.deepcopy(X_R)
        temp_yr = copy.deepcopy(y_r)
        Xt = temp_xr.pop(subject)
        yt = temp_yr.pop(subject)
        temp_xr.extend(X_L)
        temp_yr.extend(y_l)

        Xs = copy.deepcopy(temp_xr)
        ys = copy.deepcopy(temp_yr)

        if nb_sessions > 1:
            X_transfer = copy.deepcopy(Xt[:nb_sessions])
            y_transfer = copy.deepcopy(yt[:nb_sessions])
        else:
            X_transfer = Xt.pop(session)
            y_transfer = yt.pop(session)

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')

        if nb_sessions > 1:
            X_transfer = np.concatenate(X_transfer)
            y_transfer = np.concatenate(y_transfer).astype('int16')
        else:
            X_transfer = np.array(X_transfer)
            y_transfer = np.array(y_transfer).astype('int16')

        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Compute accuracy using a Progressive Neural Network (PNN) approach
def compute_accuracy_inter_subject_pnn(X, Y, param, model, nb_sessions=1):
    if nb_sessions < 1 or nb_sessions > 5:
        raise Exception("exceeded range")

    nb_subjects = len(X[NON_BOOTSTRAP_INDEX])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs, ys, Xt, yt, X_transfer, y_transfer = None, None, None, None, None, None

        Xs = copy.deepcopy(X[NON_BOOTSTRAP_INDEX])
        ys = copy.deepcopy(Y[NON_BOOTSTRAP_INDEX])
        Xt = Xs.pop(subject)
        yt = ys.pop(subject)
        if nb_sessions > 1:
            X_transfer = copy.deepcopy(Xt[:nb_sessions])
            y_transfer = copy.deepcopy(yt[:nb_sessions])
        else:
            X_transfer = Xt.pop(0)
            y_transfer = yt.pop(0)

        for i in range(len(Xs)):
            Xs[i] = np.array([sample for session in Xs[i] for sample in session])

        for i in range(len(ys)):
            ys[i] = np.array([sample for session in ys[i] for sample in session]).astype('int16')
            ys[i][ys[i] == 2] = 0
            ys[i][ys[i] == 7] = 1
            ys[i][ys[i] == 19] = 2
            ys[i][ys[i] == 23] = 3

        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')

        if nb_sessions > 1:
            X_transfer = np.concatenate(X_transfer)
            y_transfer = np.concatenate(y_transfer).astype('int16')
        else:
            X_transfer = np.array(X_transfer)
            y_transfer = np.array(y_transfer).astype('int16')

        for y in (yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Compute accuracy for inter-subject transfer with metric learning
def compute_accuracy_inter_subject_metric(X, Y, param, model, nb_sessions=1):
    if nb_sessions < 1 or nb_sessions > 5:
        raise Exception("exceeded range")

    nb_subjects = len(X[NON_BOOTSTRAP_INDEX])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs, ys, Xt, yt, X_transfer, y_transfer = None, None, None, None, None, None

        Xs = copy.deepcopy(X[NON_BOOTSTRAP_INDEX])
        ys = copy.deepcopy(Y[NON_BOOTSTRAP_INDEX])
        Xt = Xs.pop(subject)
        yt = ys.pop(subject)
        if nb_sessions > 1:
            X_transfer = copy.deepcopy(Xt[:nb_sessions])
            y_transfer = copy.deepcopy(yt[:nb_sessions])
        else:
            X_transfer = Xt.pop(0)
            y_transfer = yt.pop(0)

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')

        if nb_sessions > 1:
            X_transfer = np.concatenate(X_transfer)
            y_transfer = np.concatenate(y_transfer).astype('int16')
        else:
            X_transfer = np.array(X_transfer)
            y_transfer = np.array(y_transfer).astype('int16')

        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        print("----Metric learning----")

        mapper = learn_space_umap(Xs, ys)
        Xs_transformed = copy.deepcopy(mapper.embedding_)
        X_transfer_transformed = copy.deepcopy(mapper.transform(X_transfer))
        Xt_transformed = copy.deepcopy(mapper.transform(Xt))

        print("----Transfer learning----")
        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs_transformed, ys, X_transfer_transformed,
                                      y_transfer, Xt_transformed, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Compute accuracy with incrementally added data
def compute_accuracy_inter_subject_increment(X, Y, param, model, incr_percentage, session=0):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    incr_X, incr_Y = get_incr_data(X[NON_BOOTSTRAP_INDEX], Y[NON_BOOTSTRAP_INDEX],
                                   incr_percentage)

    incr_X, incr_Y = shuffle_data(incr_X, incr_Y)

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs = copy.deepcopy(X[NON_BOOTSTRAP_INDEX])
        ys = copy.deepcopy(Y[NON_BOOTSTRAP_INDEX])
        incr_Xs = copy.deepcopy(incr_X)
        incr_Ys = copy.deepcopy(incr_Y)
        Xt = Xs.pop(subject)
        yt = ys.pop(subject)
        Xt.pop(session)
        yt.pop(session)
        X_transfer = incr_Xs.pop(subject).pop(session)
        y_transfer = incr_Ys.pop(subject).pop(session)

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')

        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Compute accuracy for inter-hands transfer learning experiments
def compute_accuracy_inter_hands_inter_subject_increment(X, Y, param, model, incr_percentage, session=0):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    incr_X_R, incr_Y_R = get_incr_data(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND],
                                       incr_percentage)

    incr_X_R, incr_Y_R = shuffle_data(incr_X_R, incr_Y_R)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs = copy.deepcopy(X_L)
        ys = copy.deepcopy(y_l)
        Xs.pop(subject)
        ys.pop(subject)

        Xt = copy.deepcopy(X_R[subject])
        yt = copy.deepcopy(y_r[subject])
        Xt.pop(session)
        yt.pop(session)

        X_transfer = copy.deepcopy(incr_X_R[subject][session])
        y_transfer = copy.deepcopy(incr_Y_R[subject][session])

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_inter_hands_subject_increment(X, Y, param, model, incr_percentage, session=0):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    incr_X_R, incr_Y_R = get_incr_data(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND],
                                       incr_percentage)

    incr_X_R, incr_Y_R = shuffle_data(incr_X_R, incr_Y_R)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs = copy.deepcopy(X_L[subject])
        ys = copy.deepcopy(y_l[subject])

        Xt = copy.deepcopy(X_R[subject])
        yt = copy.deepcopy(y_r[subject])
        Xt.pop(session)
        yt.pop(session)

        X_transfer = copy.deepcopy(incr_X_R[subject][session])
        y_transfer = copy.deepcopy(incr_Y_R[subject][session])

        domains = np.array([session for session in range(len(Xs)) for sample in Xs[session]])

        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_inter_subject_both_increment(X, Y, param, model, incr_percentage, session=0):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    incr_X_R, incr_Y_R = get_incr_data(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND],
                                       incr_percentage)

    incr_X_R, incr_Y_R = shuffle_data(incr_X_R, incr_Y_R)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        temp_xr = copy.deepcopy(X_R)
        temp_yr = copy.deepcopy(y_r)

        Xt = temp_xr.pop(subject)
        yt = temp_yr.pop(subject)
        Xt.pop(session)
        yt.pop(session)

        temp_xr.extend(X_L)
        temp_yr.extend(y_l)

        Xs = copy.deepcopy(temp_xr)
        ys = copy.deepcopy(temp_yr)

        incr_Xs = copy.deepcopy(incr_X_R)
        incr_Ys = copy.deepcopy(incr_Y_R)

        X_transfer = incr_Xs.pop(subject).pop(session)
        y_transfer = incr_Ys.pop(subject).pop(session)

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')

        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def increment_comparison(c_model, target_model, nb_sessions=1, r=range(10, 110, 10), session=0):
    """
    Right hand test
    """
    dataType = "TDF"
    amount = 9
    X, Y = load_dataset_r()
    for i in range(len(X) - amount):
        X_normalized = l2_normalize(X)
        x = copy.deepcopy(X_normalized)
        y = copy.deepcopy(Y)
        for a in range(amount):
            x.pop(0)
            y.pop(0)
        x.pop(i)
        y.pop(i)
        tl_results = []
        c_results = []

        name = "percentage_increment_comparison_TDF_fig_IT_TRANSFER_INTER_SUBJECT_R.png"

        for incr_percentage in r:
            print("Progress:", incr_percentage)

            temp_c_result = transfer_learning_tests((x, None), (y, None), c_model, fig_name=name,
                                                    transfer_type=TRANSFER_INTER_SUBJECT_PERCENTAGE_INCREMENT,
                                                    increment=incr_percentage, nb_sessions=nb_sessions, session=session)
            temp_tl_result = transfer_learning_tests((x, None), (y, None), target_model, fig_name=name,
                                                     transfer_type=TRANSFER_INTER_SUBJECT_PERCENTAGE_INCREMENT,
                                                     increment=incr_percentage, nb_sessions=nb_sessions,
                                                     session=session)
            tl_results.append(temp_tl_result)
            c_results.append(temp_c_result)
        save_increment_comparison_results(tl_results, c_results, [*r])


def increment_comparison_both(c_model, target_model, nb_sessions=1, r=range(10, 110, 10), session=0):
    """
    Right hand test
    """
    dataType = "TDF"
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    x = copy.deepcopy(X_normalized)
    y = copy.deepcopy(Y)

    tl_results = []
    c_results = []

    name = "percentage_increment_comparison_both_TDF_fig_IT_TRANSFER_INTER_SUBJECT_R.png"

    for incr_percentage in r:
        print("Progress:", incr_percentage)

        temp_c_result = transfer_learning_tests((x, None), (y, None), c_model, fig_name=name,
                                                transfer_type=TRANSFER_INTER_SUBJECT_BOTH_PERCENTAGE_INCREMENT,
                                                increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        temp_tl_result = transfer_learning_tests((x, None), (y, None), target_model, fig_name=name,
                                                 transfer_type=TRANSFER_INTER_SUBJECT_BOTH_PERCENTAGE_INCREMENT,
                                                 increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        tl_results.append(temp_tl_result)
        c_results.append(temp_c_result)
    save_increment_comparison_results(tl_results, c_results, [*r])


def increment_comparison_inter_hands_inter_subject(c_model, target_model, nb_sessions=1, r=range(10, 110, 10),
                                                   session=0):
    """
    Right hand test
    """
    dataType = "TDF"
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    x = copy.deepcopy(X_normalized)
    y = copy.deepcopy(Y)

    tl_results = []
    c_results = []

    name = "percentage_increment_comparison_TDF_fig_IT_TRANSFER_INTER_HANDS_INTER_SUBJECT_R.png"

    for incr_percentage in r:
        print("Progress:", incr_percentage)

        temp_c_result = transfer_learning_tests((x, None), (y, None), c_model, fig_name=name,
                                                transfer_type=TRANSFER_INTER_HANDS_INTER_SUBJECT_PERCENTAGE_INCREMENT,
                                                increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        temp_tl_result = transfer_learning_tests((x, None), (y, None), target_model, fig_name=name,
                                                 transfer_type=TRANSFER_INTER_HANDS_INTER_SUBJECT_PERCENTAGE_INCREMENT,
                                                 increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        tl_results.append(temp_tl_result)
        c_results.append(temp_c_result)
    save_increment_comparison_results(tl_results, c_results, [*r])


def increment_comparison_inter_hands_subject(c_model, target_model, nb_sessions=1, r=range(10, 110, 10), session=0):
    """
    Right hand test
    """
    dataType = "TDF"
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    x = copy.deepcopy(X_normalized)
    y = copy.deepcopy(Y)

    tl_results = []
    c_results = []

    name = "percentage_increment_comparison_TDF_fig_IT_TRANSFER_INTER_HANDS_SUBJECT_R.png"

    for incr_percentage in r:
        print("Progress:", incr_percentage)

        temp_c_result = transfer_learning_tests((x, None), (y, None), c_model, fig_name=name,
                                                transfer_type=TRANSFER_INTER_HANDS_SUBJECT_PERCENTAGE_INCREMENT,
                                                increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        temp_tl_result = transfer_learning_tests((x, None), (y, None), target_model, fig_name=name,
                                                 transfer_type=TRANSFER_INTER_HANDS_SUBJECT_PERCENTAGE_INCREMENT,
                                                 increment=incr_percentage, nb_sessions=nb_sessions, session=session)
        tl_results.append(temp_tl_result)
        c_results.append(temp_c_result)
    save_increment_comparison_results(tl_results, c_results, [*r])


def subject_increment_comparison(c_model, target_model, nb_sessions=1):
    """
    Right hand test
    """
    dataType = "TDF"
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)

    tl_results = []
    c_results = []

    nb_subjects = len(X)

    name = "subject_increment_comparison_TDF_fig_IT_TRANSFER_INTER_SUBJECT_SUBJECT_R.png"
    r = range(2, nb_subjects)

    for incr_subject in r:
        x = copy.deepcopy(X_normalized)[:nb_subjects]
        y = copy.deepcopy(Y)[:nb_subjects]
        print("Progress:", incr_subject)

        temp_c_result = transfer_learning_tests((x, None), (y, None), c_model, fig_name=name,
                                                transfer_type=TRANSFER_INTER_SUBJECT_SUBJECT_INCREMENT,
                                                increment=incr_subject, nb_sessions=nb_sessions)
        temp_tl_result = transfer_learning_tests((x, None), (y, None), target_model, fig_name=name,
                                                 transfer_type=TRANSFER_INTER_SUBJECT_SUBJECT_INCREMENT,
                                                 increment=incr_subject, nb_sessions=nb_sessions)
        tl_results.append(temp_tl_result)
        c_results.append(temp_c_result)
    save_increment_comparison_results(tl_results, c_results, [*r])


def compute_accuracy_session_inter_subject(X, Y, param, model, subject=0):
    nb_sessions = 1
    accuracy_subject_tl_vec = np.array([None] * nb_sessions)
    compute_time_tl_vec = np.array([None] * nb_sessions)

    for session in range(nb_sessions):
        print("subject:", subject, "session:", session)
        Xs = copy.deepcopy(X)
        ys = copy.deepcopy(Y)

        Xt = Xs.pop(subject)
        yt = ys.pop(subject)

        X_transfer = Xt.pop(session)
        y_transfer = yt.pop(session)

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[session], \
        compute_time_tl_vec[session] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_session_subject(X, Y, param, model, subject=0):
    nb_sessions = 1
    accuracy_subject_tl_vec = np.array([None] * nb_sessions)
    compute_time_tl_vec = np.array([None] * nb_sessions)

    for session in range(nb_sessions):
        print("subject:", subject, "session:", session)
        Xs = copy.deepcopy(X)
        ys = copy.deepcopy(Y)

        Xt = Xs.pop(subject)
        yt = ys.pop(subject)

        Xs = Xt.pop(session)
        ys = yt.pop(session)

        next_session = session % len(Xt)  # Xt got smaller, session here is the next session

        X_transfer = Xt.pop(next_session)
        y_transfer = yt.pop(next_session)

        Xs = np.array(Xs)
        ys = np.array(ys).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[session], \
        compute_time_tl_vec[session] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_subject_subject(X, Y, param, model, subject=0):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX])
    session = 0
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    for it_subject in range(nb_subjects):
        print("subject:", it_subject, "session:", session)
        Xs = copy.deepcopy(X[NON_BOOTSTRAP_INDEX])
        ys = copy.deepcopy(Y[NON_BOOTSTRAP_INDEX])

        Xt = Xs[subject]
        yt = ys[subject]

        Xs = Xs[it_subject]
        ys = ys[it_subject]

        X_transfer = Xt.pop(session)
        y_transfer = yt.pop(session)

        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[it_subject], \
        compute_time_tl_vec[it_subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_inter_hands_subject(X, Y, param, model):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs = copy.deepcopy(X_L[subject])
        ys = copy.deepcopy(y_l[subject])

        Xt = copy.deepcopy(X_R[subject])
        yt = copy.deepcopy(y_r[subject])

        X_transfer = Xt.pop(0)
        y_transfer = yt.pop(0)

        domains = np.array([session for session in range(len(Xs)) for sample in Xs[session]])

        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


def compute_accuracy_inter_hands_inter_subject(X, Y, param, model):
    nb_subjects = len(X[NON_BOOTSTRAP_INDEX][RIGHT_HAND])
    accuracy_subject_tl_vec = np.array([None] * nb_subjects)
    compute_time_tl_vec = np.array([None] * nb_subjects)

    X_R, X_L, y_r, y_l = X[NON_BOOTSTRAP_INDEX][RIGHT_HAND], X[NON_BOOTSTRAP_INDEX][LEFT_HAND], \
                         Y[NON_BOOTSTRAP_INDEX][RIGHT_HAND], Y[NON_BOOTSTRAP_INDEX][LEFT_HAND]

    for subject in range(nb_subjects):
        print("subject:", subject)
        Xs = copy.deepcopy(X_L)
        ys = copy.deepcopy(y_l)
        Xs.pop(subject)
        ys.pop(subject)

        Xt = copy.deepcopy(X_R[subject])
        yt = copy.deepcopy(y_r[subject])

        X_transfer = Xt.pop(0)
        y_transfer = yt.pop(0)

        domains = np.array([i for i in range(len(Xs)) for session in Xs[i] for sample in session])

        Xs = np.array([sample for subject in Xs for session in subject for sample in session])
        ys = np.array([sample for subject in ys for session in subject for sample in session]).astype('int16')
        Xt = np.concatenate(Xt)
        yt = np.concatenate(yt).astype('int16')
        X_transfer = np.array(X_transfer)
        y_transfer = np.array(y_transfer).astype('int16')
        for y in (ys, yt, y_transfer):
            y[y == 2] = 0
            y[y == 7] = 1
            y[y == 19] = 2
            y[y == 23] = 3

        accuracy_subject_tl_vec[subject], \
        compute_time_tl_vec[subject] \
            = compute_accuracy[model](Xs, ys, X_transfer, y_transfer, Xt, yt, param, domains)

    return accuracy_subject_tl_vec, \
           compute_time_tl_vec


# Conducts transfer learning tests and evaluations based on specified parameters and scenarios
def transfer_learning_tests(X, Y, model, fig_name="exp_fig.png", transfer_type=TRANSFER_INTER_SUBJECT, subject=0,
                            increment=None, nb_sessions=1, session=0):
    m_params = model_params[model]
    accuracy_per_param_tl_mean = pandas.array([None] * len(m_params))
    accuracy_per_param_tl = [None] * len(m_params)
    compute_time_per_param_tl_mean = pandas.array([None] * len(m_params))
    compute_time_per_param_tl = [None] * len(m_params)

    for i in range(len(m_params)):
        if transfer_type == TRANSFER_INTER_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject(X, Y, m_params[i], model, nb_sessions, session=session)
        elif transfer_type == TRANSFER_INTER_SUBJECT_BOTH:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject_both(X, Y, m_params[i], model, nb_sessions, session=session)
        elif transfer_type == TRANSFER_SESSION_INTER_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_session_inter_subject(X, Y, m_params[i], model, subject=subject)
        elif transfer_type == TRANSFER_SESSION_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_session_subject(X, Y, m_params[i], model, subject=subject)
        elif transfer_type == TRANSFER_INTER_HANDS_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_hands_subject(X, Y, m_params[i], model)
        elif transfer_type == TRANSFER_INTER_HANDS_INTER_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_hands_inter_subject(X, Y, m_params[i], model)
        elif transfer_type == TRANSFER_INTER_SUBJECT_PERCENTAGE_INCREMENT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject_increment(X, Y, m_params[i], model, incr_percentage=increment,
                                                         session=session)
        elif transfer_type == TRANSFER_INTER_SUBJECT_BOTH_PERCENTAGE_INCREMENT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject_both_increment(X, Y, m_params[i], model, incr_percentage=increment,
                                                              session=session)
        elif transfer_type == TRANSFER_SUBJECT_SUBJECT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_subject_subject(X, Y, m_params[i], model, subject=subject)
        elif transfer_type == TRANSFER_INTER_SUBJECT_METRIC:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject_metric(X, Y, m_params[i], model, nb_sessions)
        elif transfer_type == TRANSFER_INTER_SUBJECT_PNN:
            if model != get_model("PNN"):
                raise ValueError("Only pnn allowed")
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject_pnn(X, Y, m_params[i], model, nb_sessions)
        elif transfer_type == TRANSFER_INTER_SUBJECT_SUBJECT_INCREMENT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_subject(X, Y, m_params[i], model, nb_sessions)
        elif transfer_type == TRANSFER_INTER_HANDS_INTER_SUBJECT_PERCENTAGE_INCREMENT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_hands_inter_subject_increment(X, Y, m_params[i], model,
                                                                     incr_percentage=increment, session=session)
        elif transfer_type == TRANSFER_INTER_HANDS_SUBJECT_PERCENTAGE_INCREMENT:
            accuracy_per_param_tl[i], \
            compute_time_per_param_tl[i] = \
                compute_accuracy_inter_hands_subject_increment(X, Y, m_params[i], model, incr_percentage=increment,
                                                               session=session)
        else:
            raise Exception

        accuracy_per_param_tl_mean[i], \
        compute_time_per_param_tl_mean[i] = \
            accuracy_per_param_tl[i].mean(), \
            compute_time_per_param_tl[i].mean()

        print("mean", accuracy_per_param_tl_mean[i])

    return X, model, accuracy_per_param_tl, \
           accuracy_per_param_tl_mean, compute_time_per_param_tl, \
           compute_time_per_param_tl_mean, \
           fig_name, m_params, subject, transfer_type
