This repository contains a collection of machine learning assignments completed as part of an introductory course. The projects explore various machine learning concepts, algorithms, model evaluation techniques, and important considerations like data bias.

The assignments cover:
1.  Text Classification using different models and text vectorization techniques.
2.  Classification with various algorithms (Decision Trees, Naive Bayes, SVM) and hyperparameter tuning, including a custom SVM implementation.
3.  Building a Naive Bayes classifier from scratch and analyzing bias in a real-world risk assessment dataset.

## Project Descriptions

### Assignment 1: Text Classification with Various Machine Learning Models (PA1.ipynb)

* **Objective:** Learn and apply fundamental machine learning concepts and models for text classification using the "20 Newsgroups" dataset.
* **Key Activities:**
    * Implemented and compared Linear Models (Logistic Regression), Decision Trees, and Neural Networks for multi-class text classification.
    * Utilized Bag of Words (BoW) for text vectorization, including 1-grams, 2-grams, stop-word removal, and frequency filtering (max_df=0.8, min_df=0.001), resulting in a feature space of **21,231 dimensions**.
    * Evaluated model performance on a test set using Accuracy, Precision, Recall, and F1 Score (macro averaging). Achieved **0.792 Accuracy** with Logistic Regression and **0.827 Accuracy** with a two-hidden-layer Neural Network.
    * Analyzed the most influential features (words/2-grams) for the Logistic Regression model based on learned coefficients.

### Assignment 2: Classification and Bias Analysis on Healthcare and Criminal Justice Data (PA2.ipynb)

* **Objective:** Practice implementing and evaluating various classification algorithms and explore data bias using healthcare and criminal justice datasets.
* **Key Activities:**
    * Applied Decision Trees, Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes classifiers to the Breast Cancer Wisconsin (Diagnostic) dataset.
    * Explored hyperparameter tuning for the Decision Tree using GridSearchCV (`max_depth` and `min_samples_leaf`) to mitigate overfitting.
    * Implemented a core Support Vector Machine (SVM) algorithm from scratch using `CVXPY` for optimization, including data scaling with `StandardScaler`.
    * Evaluated the custom SVM model, reporting **0.956 Accuracy**, **0.961 Macro Precision**, **0.945 Macro Recall**, and **0.952 Macro F1 Score** on the test set.
    * Initiated an exploration into data bias by analyzing False Positive Rate (FPR) and False Negative Rate (FNR) across different racial groups in a COMPAS-style risk assessment context, calculating metrics such as **African-American FPR: 0.95** and **Asian FPR: 0.00**, indicating potential disparities.

### Assignment 3: Building a Naïve Bayes Classifier and Exploring Data Bias (PA3_hbasavar.ipynb)

* **Objective:** Deepen understanding of the Naïve Bayes algorithm by implementing it manually and investigate the impact of data characteristics on model bias in a predictive system inspired by criminal justice tools.
* **Key Activities:**
    * Implemented a Gaussian Naïve Bayes classifier from scratch, including calculating prior probabilities, feature means and variances, likelihoods, and posterior probabilities based on Bayes' Theorem.
    * Validated the custom Naïve Bayes implementation by comparing its accuracy to the scikit-learn library's `GaussianNB` on the Iris dataset, achieving **100% accuracy** for both.
    * Applied a Gaussian Naïve Bayes model to a COMPAS recidivism dataset and performed a bias analysis by calculating False Positive Rate (FPR) and False Negative Rate (FNR) specifically for different racial categories to identify potential fairness issues.
    * Visualized and discussed the observed differences in FPR and FNR across racial groups, noting disparities that suggest bias in the model's risk predictions.

## Running the Code

To run the Jupyter notebooks in this repository, you'll need to have Python and the necessary libraries installed.

1.  Clone the repository:
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    ```
2.  Navigate to the repository directory:
    ```bash
    cd [your-repository-name]
    ```
3.  (Optional but Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    ```
    * On Windows: `venv\Scripts\activate`
    * On macOS/Linux: `source venv/bin/activate`
4.  Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
5.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
    This will open a tab in your web browser where you can open and run the `.ipynb` files.
