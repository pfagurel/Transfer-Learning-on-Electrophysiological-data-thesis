# mainTransferClassification.py
# Description: Main file for running the transfer learning experiments.

from data_visualization import *
from experiments import *
from pad_calculation import *

np.random.seed(1)


def test_inter_hands_subject_right(model):
    X, Y = load_dataset_both()
    save_results(
        *transfer_learning_tests((X, None), (Y, None), model, fig_name="TDF_fig_TRANSFER_INTER_HANDS_SUBJECT_R.png",
                                 transfer_type=TRANSFER_INTER_HANDS_SUBJECT))


def test_inter_hands_subject_right_l2_norm(model):
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_INTER_HANDS_SUBJECT_R_l2_norm.png",
                                          transfer_type=TRANSFER_INTER_HANDS_SUBJECT))


def test_inter_hands_inter_subject_right(model):
    X, Y = load_dataset_both()
    save_results(*transfer_learning_tests((X, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_INTER_HANDS_INTER_SUBJECT_R.png",
                                          transfer_type=TRANSFER_INTER_HANDS_INTER_SUBJECT))


def test_inter_hands_inter_subject_right_l2_norm(model):
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_INTER_HANDS_INTER_SUBJECT_R_l2_norm.png",
                                          transfer_type=TRANSFER_INTER_HANDS_INTER_SUBJECT))


def test_inter_subject_right(model, nb_sessions=1):
    X, Y = load_dataset_r()
    save_results(*transfer_learning_tests((X, None), (Y, None), model, fig_name="nb_sess " + str(
        nb_sessions) + " TDF_fig_TRANSFER_INTER_SUBJECT_R.png", transfer_type=TRANSFER_INTER_SUBJECT,
                                          nb_sessions=nb_sessions))


def test_inter_subject_right_l2_norm(model, nb_sessions=1, session=0):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model, fig_name="nb_sess " + str(
        nb_sessions) + " session " + str(session) + " TDF_fig_TRANSFER_INTER_SUBJECT_R_l2_norm.png",
                                          transfer_type=TRANSFER_INTER_SUBJECT, nb_sessions=nb_sessions,
                                          session=session))


def test_inter_subject_right_pnn_l2_norm(nb_sessions=1):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), get_model("PNN"), fig_name="nb_sess " + str(
        nb_sessions) + " TDF_fig_TRANSFER_INTER_SUBJECT_R_l2_norm.png", transfer_type=TRANSFER_INTER_SUBJECT_PNN,
                                          nb_sessions=nb_sessions))


def test_inter_subject_both_l2_norm(model):
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_INTER_SUBJECT_BOTH_l2_norm.png",
                                          transfer_type=TRANSFER_INTER_SUBJECT_BOTH))


def test_session_inter_subject_both_l2_norm(model):
    X, Y = load_dataset_both()
    X_normalized = l2_normalize(X[RIGHT_HAND]), l2_normalize(X[LEFT_HAND])
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_SESSION_INTER_SUBJECT_BOTH.png",
                                          transfer_type=TRANSFER_SESSION_INTER_SUBJECT_BOTH, subject=0))


def test_session_inter_subject_right_l2_norm(model):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_SESSION_INTER_SUBJECT_R.png",
                                          transfer_type=TRANSFER_SESSION_INTER_SUBJECT, subject=0))


def test_session_subject_right_l2_norm(model):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="TDF_fig_TRANSFER_SESSION_SUBJECT_R.png",
                                          transfer_type=TRANSFER_SESSION_SUBJECT, subject=0))


def test_subject_subject_right_l2_norm(model, subject):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model,
                                          fig_name="subject " + str(subject) + "TDF_fig_TRANSFER_SUBJECT_SUBJECT_R.png",
                                          transfer_type=TRANSFER_SUBJECT_SUBJECT, subject=subject))


def test_inter_subject_right_metric_l2_norm(model, nb_sessions=1):
    X, Y = load_dataset_r()
    X_normalized = l2_normalize(X)
    save_results(*transfer_learning_tests((X_normalized, None), (Y, None), model, fig_name="nb_sess " + str(
        nb_sessions) + " TDF_fig_TRANSFER_INTER_SUBJECT_METRIC_R_l2_norm.png",
                                          transfer_type=TRANSFER_INTER_SUBJECT_METRIC, nb_sessions=nb_sessions))


if __name__ == "__main__":

    # Load right-hand movement dataset and apply L2 normalization
    X, Y = load_dataset_r()
    X = l2_normalize(X)
    l2_X = l2_normalize(X)

    # Calculate Proxy A-Distance (PAD) within the same subject (intra-subject)
    print(calculate_pad_intra_subject(l2_X, 2))

    # Calculate Proxy A-Distance (PAD) between different subjects (inter-subject)
    print(calculate_pad_inter_subject(l2_X, 0))

    # Visualize data using UMAP and PCA for dimensionality reduction
    visualise_data("umap", None, l2_X, Y)
    visualise_data("pca", None, l2_X, Y)

    # Visualize data per subject using UMAP and PCA
    visualise_data_per_subject("umap", None, l2_X, Y)
    visualise_data_per_subject("pca", None, l2_X, Y)

    # Evaluate various transfer learning models on the right-hand dataset with L2 normalization
    test_inter_subject_right_pnn_l2_norm()
    test_inter_subject_right_l2_norm(get_model("Linear_TgtOnly"))
    test_inter_subject_right(get_model("CORAL_linear"), nb_sessions=5)
    test_inter_subject_both_l2_norm(get_model("SA_linear"))
    test_inter_subject_right_l2_norm(get_model("FA_linear"))

    # Evaluate models for inter-hand and inter-subject scenarios
    test_inter_hands_inter_subject_right_l2_norm("TrAdaBoost_linear")
    test_inter_hands_subject_right_l2_norm(get_model("FA_linear"))

    # Assess SA linear model with metric learning for right hand
    test_inter_subject_right_metric_l2_norm(get_model("SA_linear"))

    # Re-evaluate FA linear model for the right hand, focusing on session 1
    test_inter_subject_right_l2_norm(get_model("FA_linear"), session=1)

    # Perform incremental data addition comparisons
    increment_comparison_both(get_model("Linear_TgtOnly"), get_model("Linear_All"), r=range(10, 100, 10), session=0)
    increment_comparison_inter_hands_subject(get_model("Linear_TgtOnly"), get_model("Linear_All"))
    increment_comparison(get_model("Linear_TgtOnly"), get_model("FA_linear"), r=range(10, 100, 10), session=0)



