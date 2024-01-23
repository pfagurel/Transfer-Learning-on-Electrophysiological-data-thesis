# pad_calculation.py
# Description: This file provides functions to calculate the Proxy A-Distance (PAD) between different domains,
# sessions, or subjects within a dataset. PAD is a measure of domain discrepancy or the difficulty of transferring
# knowledge from one domain to another. It utilizes a RandomForestClassifier to estimate the error rate of
# distinguishing between two domains. Functions include calculating PAD within a subject across sessions, between
# different subjects, and between hands within a subject.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np


# Calculate PAD between two domains
def calculate_pad(domain1, domain2):
    # Define the labels
    labels1 = [0] * len(domain1)  # Labels for domain1
    labels2 = [1] * len(domain2)  # Labels for domain2

    # Combine the data from the two domains
    X = np.concatenate((domain1, domain2))
    y = labels1 + labels2

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute the error rate
    error_rate = mean_absolute_error(y_test, y_pred)
    print(error_rate)
    # Compute the Proxy A-Distance (PAD)
    pad = 2 * (1 - 2 * error_rate)

    return pad


# Calculate PAD within a subject across different sessions
def calculate_pad_intra_subject(X, subject_idx):
    num_sessions = len(X[subject_idx])
    pad_values = []
    for session_idx in range(1, num_sessions):  # Start from 1 because we are comparing with session 0
        pad = calculate_pad(X[subject_idx][0], X[subject_idx][session_idx])
        pad_values.append((max(0, pad) * 100) / 2)
    return pad_values


# Calculate PAD between the specified subject and all other subjects
def calculate_pad_inter_subject(X, subject_idx):
    num_subjects = len(X)
    subject_data = np.concatenate(X[subject_idx])
    pad_values = []
    for subject_idx in range(num_subjects):
        pad = calculate_pad(subject_data, np.concatenate(X[subject_idx]))
        pad_values.append((max(0, pad) * 100) / 2)
    return pad_values


# Calculate PAD between right and left hand data within a subject
def calculate_pad_inter_hands_intra_subject(X_r, X_l, subject_idx):
    num_sessions = len(X_l[subject_idx])
    pad_values = []
    for session_idx in range(num_sessions):  # Start from 1 because we are comparing with session 0
        pad = calculate_pad(X_r[subject_idx][0], X_l[subject_idx][session_idx])
        pad_values.append((max(0, pad) * 100) / 2)
    return pad_values