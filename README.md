# Transfer Learning on Electrophysiological data : Master's Thesis

This repository contains the implementation and documentation of a transfer learning on EMG data, as part of my Master's Thesis.

![Local Image](https://github.com/pfagurel/Transfer-Learning-on-Electrophysiological-data-thesis/blob/main/Problem_illustration.png)

_"Gesture estimation in the inter-subject setting is hard to solve because the electrophysiological signals vary significantly between different subjects. This can be due to differences in muscle structure, movement patterns, skin conductivity, and other physiological factors. This high degree of variability makes it challenging for a model to generalize its predictions to new, unseen subjects._
 
_To tackle this problem we would want to take advantage of the previously acquired training "knowledge". The field of machine learning tackling this problem is called Transfer Learning."_
 
The thesis presents findings from the application of Transfer Learning algorithms to different scenarios, such as:
 
*	single hand data, 
*	both hands data, 
*	between hands and 
*	unsupervised Transfer Learning.
 
In the end, it shows that applying Transfer Learning, indeed improves the performance in all those settings. 

## Repository Contents

- **`constants.py`**: Constants used across the project for model configurations and experiment setups.
- **`dataset/`**: Directory containing datasets utilized in training and evaluating the models.
- **`data_visualization.py`**: Scripts for visualizing the gesture data using PCA, UMAP, and other techniques.
- **`experiments.py`**: Detailed implementation of transfer learning experiments, including intra-subject and inter-subject transfer.
- **`mainTransferClassification.py`**: Main script to run transfer learning experiments and generate results.
- **`MastersThesis_PetroFagurel.pdf`**: The complete Master's Thesis document outlining the research, methodology, results, and conclusions.
- **`model_evaluation.py`**: Functions for assessing the performance of various transfer learning models.
- **`model_parameters.py`**: Model parameters and configurations for the experiments.
- **`pad_calculation.py`**: Functions to calculate the Proxy A-Distance (PAD) for domain adaptation analysis.
- **`PNN.py`**: Implementation of the Progressive Neural Network (PNN) model for transfer learning.
- **`utils.py`**: Utility functions for data preprocessing, normalization, and miscellaneous operations.

## Getting Started

### Prerequisites

- Python 3.9

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/pfagurel/Transfer-Learning-on-Electrophysiological-data-thesis
```
Install the required Python packages:
```bash
pip install adapt matplotlib pandas seaborn numpy scikit-learn umap-learn
```
### Usage

Execute the main script to run experiments:

```bash
python mainTransferClassification.py
```

