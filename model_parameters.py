# model_parameters.py
# Description: This file defines the mapping of model names to their respective accuracy computation functions and parameters.
# It serves as a configuration file for accessing the various machine learning and transfer learning models used in the project.
# The module includes a dictionary mapping model names to functions defined in 'model_evaluation.py'
# and another dictionary specifying parameters for each model, such as the number of trees in a random forest or the number of neurons in an MLP.

from model_evaluation import *

# Mapping of model names to their corresponding accuracy computation functions
compute_accuracy = {"RELAB": compute_accuracy_relab,
                    "STRUT": compute_accuracy_strut,
                    "SER": compute_accuracy_ser,
                    "TrAdaBoost_rf": compute_accuracy_tab_rf,
                    "TrAdaBoost_linear": compute_accuracy_tab_linear,
                    "TrAdaBoost_mlp": compute_accuracy_tab_mlp,
                    "SA_rf": compute_accuracy_sa_rf,
                    "SA_linear": compute_accuracy_sa_linear,
                    "SA_mlp": compute_accuracy_sa_mlp,
                    "CORAL_rf": compute_accuracy_coral_rf,
                    "CORAL_linear": compute_accuracy_coral_linear,
                    "CORAL_mlp": compute_accuracy_coral_mlp,
                    "KLIEP_rf": compute_accuracy_kliep_rf,
                    "KLIEP_linear": compute_accuracy_kliep_linear,
                    "KLIEP_mlp": compute_accuracy_kliep_mlp,
                    "RF_SrcOnly": compute_accuracy_rf_srconly,
                    "RF_TgtOnly": compute_accuracy_rf_tgtonly,
                    "RF_All": compute_accuracy_rf_all,
                    "Linear_SrcOnly": compute_accuracy_linear_srconly,
                    "Linear_TgtOnly": compute_accuracy_linear_tgtonly,
                    "Linear_All": compute_accuracy_linear_all,
                    "MLP_SrcOnly": compute_accuracy_mlp_srconly,
                    "MLP_TgtOnly": compute_accuracy_mlp_tgtonly,
                    "MLP_All": compute_accuracy_mlp_all,
                    "PNN": compute_accuracy_pnn,
                    "FA_rf": compute_accuracy_fa_rf,
                    "FA_linear": compute_accuracy_fa_linear,
                    "FA_mlp": compute_accuracy_fa_mlp,
                    "FA_svm": compute_accuracy_fa_svm,
                    "FA_hgb": compute_accuracy_fa_hgb,
                    }


# Dictionary defining specific parameters for each model
model_params = {"RELAB": [50],  # trees
                "STRUT": [50],  # trees
                "SER": [50],  # trees
                "TrAdaBoost_rf": [50],  # RF trees
                "TrAdaBoost_linear": [None],
                "TrAdaBoost_mlp": [10],  # mlp hidden layer neurons
                "SA_rf": [50],  # trees
                "SA_linear": [None],
                "SA_mlp": [10],  # mlp hidden layer neurons
                "CORAL_rf": [50],  # trees
                "CORAL_linear": [None],
                "CORAL_mlp": [10],  # mlp hidden layer neurons
                "KLIEP_rf": [50],  # trees
                "KLIEP_linear": [None],
                "KLIEP_mlp": [10],  # mlp hidden layer neurons
                "RF_SrcOnly": [50],  # trees
                "RF_TgtOnly": [50],  # trees
                "RF_All": [50],  # trees
                "Linear_SrcOnly": [None],
                "Linear_TgtOnly": [None],
                "Linear_All": [None],
                "MLP_SrcOnly": [10],  # mlp hidden layer neurons
                "MLP_TgtOnly": [10],  # mlp hidden layer neurons
                "MLP_All": [10],  # mlp hidden layer neurons
                "PNN": [10],  # neurons for each task's hidden layer
                "FA_rf": [50],  # trees
                "FA_linear": [None],
                "FA_mlp": [10],  # mlp hidden layer neurons
                "FA_svm": [None],
                "FA_hgb": [50]
                }


# Function to validate and return the model name
def get_model(model_name):
    if model_name in model_params.keys():
        return model_name
    raise Exception