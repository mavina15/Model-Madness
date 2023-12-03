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

