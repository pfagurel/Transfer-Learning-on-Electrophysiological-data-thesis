# constants.py
# Description: This file defines a set of constants used throughout the project.
# These constants primarily represent different types of transfer learning scenarios
# and configurations, aiding in categorizing and managing various experiment setups.
# Additionally, constants for data indexing like hand type and bootstrap methods are included.

# Transfer learning experiments constants
TRANSFER_INTER_SUBJECT = 0
TRANSFER_INTER_SUBJECT_BOTH = 1
TRANSFER_INTER_SUBJECT_METRIC = 2
TRANSFER_INTER_SUBJECT_PNN = 3
TRANSFER_INTER_SUBJECT_PERCENTAGE_INCREMENT = 4
TRANSFER_SESSION_INTER_SUBJECT = 5
TRANSFER_SESSION_INTER_SUBJECT_BOTH = 6
TRANSFER_SESSION_SUBJECT = 7
TRANSFER_INTER_HANDS_SUBJECT = 8
TRANSFER_INTER_HANDS_INTER_SUBJECT = 9
TRANSFER_INTER_SUBJECT_SUBJECT_INCREMENT = 10
TRANSFER_SUBJECT_SUBJECT = 11
TRANSFER_INTER_SUBJECT_BOTH_PERCENTAGE_INCREMENT = 12
TRANSFER_INTER_HANDS_INTER_SUBJECT_PERCENTAGE_INCREMENT = 13
TRANSFER_INTER_HANDS_SUBJECT_PERCENTAGE_INCREMENT = 14

# Indexing constants for different data configurations
NON_BOOTSTRAP_INDEX = 0
BOOTSTRAP_INDEX = 1

# Constants representing left and right hands in datasets
RIGHT_HAND = 0
LEFT_HAND = 1
