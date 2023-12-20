# Comparing 10 different machine learning models to find the best one for breast cancer classification

## To replicate: 
1. Download .ipynb
2. Upload to
3. Run!

## Logistic Regression
Logistic Regression is a machine learning model that is good for categorizing data. 

Results from the notebook:

Model: Logistic Regression


Confusion Matrix:
[ 62   1]
[  2 106]


Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98        63
           1       0.99      0.98      0.99       108

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171



AUC Score: 0.9980893592004703

Logistic Regression has a precision of 0.97, a recall of 0.98 and an f1 score of 0.98.

## K Nearest Neighbors
K Nearest Neighbors is a machine learning model that is good for XYZ. 

Model: KNeighborsClassifier


Confusion Matrix:
[[ 59   4]
 [  3 105]]


Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.94      0.94        63
           1       0.96      0.97      0.97       108

    accuracy                           0.96       171
   macro avg       0.96      0.95      0.96       171
weighted avg       0.96      0.96      0.96       171



AUC Score: 0.9776601998824221

K Nearest Neighbors has a precision of 0.97, a recall of 0.98 and an f1 score of 0.98.

Final Rankings by F1:
1
2
3

Final Rankings by AUC:
1
2
3

ROC Visual
![image](https://github.com/mavina15/modelmadness/assets/11577013/fdbd7491-ae37-408c-aef1-3779f228cab4)

Preciscion-Recall Visual
![image](https://github.com/mavina15/modelmadness/assets/11577013/84278c83-0335-4d79-ac72-2f4d5b549950)

# Model Madness: A Deep Dive into 10 Machine Learning Models

## Introduction
Welcome to Model Madness! In this repository, we explore the performance of 10 different machine learning models using the Breast Cancer dataset. The goal is to predict whether an individual has breast cancer based on various features. This README provides an overview of the code, model descriptions, and visualizations included in the project.

## Getting Started
To run the code, ensure you have the required libraries installed. You can install them using the following:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## Code Overview
The code is organized into sections:

1. **Importing Libraries:** Importing necessary libraries for data processing and model evaluation.

2. **Loading and Preprocessing Data:** Loading the Breast Cancer dataset, reformatting the data, and splitting it into training and testing sets.

3. **Evaluation Function:** Defining a function (`evaluate`) to assess model performance, including confusion matrix, classification report, and AUC score.

4. **Models:** Implementation of 10 machine learning models:
   - Logistic Regression
   - K-Nearest Neighbor
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Naive Bayes
   - Neural Networks (MLP)
   - Ada Boost
   - XG Boost

5. **Model Descriptions:** Brief descriptions of each machine learning model.

6. **ROC Curves Visualization:** Visualization of ROC curves for selected models.

7. **Precision-Recall Curves Visualization:** Visualization of Precision-Recall curves for selected models.

## Model Descriptions
1. **Logistic Regression:** Predicts outcomes using mathematical relationships between data factors.
   
2. **K-Nearest Neighbor:** A non-parametric method that approximates associations between variables by averaging observations in the same neighborhood.

3. **Support Vector Machine (SVM):** A linear model for classification and regression problems, creating a line or hyperplane to separate data into classes.

4. **Decision Tree:** A graphical representation of possible solutions based on conditions.

5. **Random Forest:** An ensemble of decision trees trained with the bagging method to improve predictive accuracy.

6. **Gradient Boosting:** An algorithm that iteratively calculates points using gradients and scales steps for optimization.

7. **Naive Bayes:** Assumes the presence of one feature in a class is unrelated to the presence of any other feature.

8. **Neural Networks (MLP):** A feed-forward artificial neural network that generates outputs from inputs.

9. **Ada Boost:** An ensemble learning technique that combines multiple "weak" learners to improve predictive accuracy.

10. **XG Boost:** An implementation of gradient-boosting decision trees.

## Visualization
- **ROC Curves:** A graph showing the performance of classification models at all thresholds.

- **Precision-Recall Curves:** A metric evaluating classifier quality, especially in imbalanced classes.

Feel free to explore, analyze, and compare the performance of these machine learning models in the context of breast cancer prediction!

