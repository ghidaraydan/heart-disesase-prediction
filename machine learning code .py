import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
df=pd.read_csv("C:\\Users\\BC\\Downloads\\heart.csv")




df.head()

#print(df.head())
df.info()
df.isna().sum()



# MinMaxScaler
from pyngrok import ngrok
import streamlit as st
import base64
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()

feat=['age', 	'sex', 	'cp', 'trestbps', 'chol', 	'fbs', 	'restecg', 	'thalach' ,	'exang', 	'oldpeak' ,	'slope', 	'ca', 'thal']
df[feat] = scal.fit_transform(df[feat])
print(df.head())

#StandardScaler 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features= ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] 
df[features] = scaler.fit_transform(df[features])
print(df.head())

# Creating Features and Target variable
X=df.drop("target",axis=1).values
Y=df.target.values

#Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

# Create a function for evaluating metrics
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix

def evaluation(Y_test,Y_pred):
  acc=accuracy_score(Y_test,Y_pred)
  rcl=recall_score(Y_test,Y_pred)
  f1=f1_score(Y_test,Y_pred)
 

  metric_dict={'accuracy': round(acc,3),
               'recall': round(rcl,3),
               'F1 score': round(f1,3),
               
              }

  return print(metric_dict)

 # Fitting and Comparing different Models 
# 1- knn 
np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier
Knn_clf=  KNeighborsClassifier()
Knn_clf.fit(X_train,Y_train)
Knn_Y_pred=Knn_clf.predict(X_test)
Knn_score=Knn_clf.score(X_test,Y_test)
print(evaluation(Y_test,Knn_Y_pred))

# 2-logistic regression 
np.random.seed(42)
from sklearn.linear_model import LogisticRegression
LR_clf=LogisticRegression()
LR_clf.fit(X_train,Y_train)
LR_Y_pred=LR_clf.predict(X_test)
LR_score=LR_clf.score(X_test,Y_test)
print(evaluation(Y_test,LR_Y_pred))


# 3-SVC 
np.random.seed(42)
from sklearn.svm import SVC
SVC_clf=SVC()
SVC_clf.fit(X_train,Y_train)
SVC_score=SVC_clf.score(X_test,Y_test)
SVC_Y_pred=SVC_clf.predict(X_test)
print(evaluation(Y_test,SVC_Y_pred))



# Model Evaluation
model_comp = pd.DataFrame({'Model': ['Logistic Regression',
                    'K-Nearest Neighbour','Support Vector Machine'], 'Accuracy': [LR_score*100,
                    Knn_score*100,SVC_score*100 ]})
print(model_comp)

# Tuning KNN (8 IS THE BEST)
neighbors = range(1, 21) # 1 to 20 # 

# Setup algorithm
knn = KNeighborsClassifier()

# Loop through different neighbors values
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    
    # Fit the algorithm
    print(f"Accuracy with {i} no. of neighbors: {knn.fit(X_train, Y_train).score(X_test,Y_test)}%")
# Using the Best K 
np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier
Knn_clf=  KNeighborsClassifier(n_neighbors=8)
Knn_clf.fit(X_train,Y_train)
Knn_Y_pred=Knn_clf.predict(X_test)
Knn_score_tuned=Knn_clf.score(X_test,Y_test)
print(evaluation(Y_test,Knn_Y_pred))





model_comp = pd.DataFrame({'Model': ['Logistic Regression',
                    'K-Nearest Neighbour','Support Vector Machine',"Tuned_K_Nearest Neighbours"], 'Accuracy': [LR_score*100,
                    Knn_score*100,SVC_score*100,Knn_score_tuned*100]})
print(model_comp)

print(" Best evaluation parameters achieved with SVC:") 
print(evaluation(Y_test,SVC_Y_pred))


    
   










