# Breast Cancer Classification with SVM

This project implements a machine learning model for classifying breast cancer cases as **Malignant** or **Benign** using the **Support Vector Machine (SVM)** algorithm with hyperparameter tuning and evaluation through various metrics.

## Project Overview

The dataset used is from the UCI Machine Learning Repository and contains features extracted from digitized images of fine needle aspirates (FNA) of breast mass. The goal is to predict whether a tumor is malignant or benign based on these features.

### Key Steps in the Project

1. **Data Preprocessing**:
    - Loading and cleaning the data.
    - Handling missing values by replacing them with the mean of the respective feature.
    - Converting categorical labels (Malignant/Benign) into binary values.

2. **Feature Scaling**:
    - The features are standardized using `StandardScaler` to bring them to a common scale.

3. **Model Building**:
    - The model is built using **SVM** with **RBF kernel** and optimized using **GridSearchCV** for hyperparameter tuning (`C` and `gamma`).

4. **Model Evaluation**:
    - The model's performance is evaluated using cross-validation, accuracy score, classification report, confusion matrix, and ROC curve.

## Installation and Setup

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

Clone this repository to your local machine:

    git clone https://github.com/yourusername/breast-cancer-classification.git

### 2. Install the Dependencies

The following libraries are required:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using the following command:

    pip install -r requirements.txt

Alternatively, install the dependencies individually:

    pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Run the Project
After installing the dependencies, run the script to start training and evaluating the model:
    ```
    python breast_cancer_classification.py
This will train the model, perform hyperparameter tuning, and display the evaluation results, including the classification report, confusion matrix, and ROC curve.

## Results
The model achieves high accuracy and provides a detailed classification report. It shows precision, recall, and F1-score for both **Benign** and **Malignant** classes, along with a ROC curve and AUC score indicating the model's performance.

## Contact
- **Email**: [mehrdadkarimi79@gmail.com](mailto:mehrdadkarimi79@gmail.com)
- **Telegram**: [@TheyCallMeMehri](https://t.me/TheyCallMeMehri)
