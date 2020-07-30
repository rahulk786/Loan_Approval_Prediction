#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:51:23 2020

@author: vimal
"""

#Importing the library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

#Importing the dataset
df = pd.read_csv('Proj 4-Train Data.csv')
df_1=pd.read_csv('Proj 4-CleanTestData.csv')

#Data Preprocessing
#Removing unwanted columns
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
df=df.dropna(axis=0, subset=['Loanapp_ID'])
df.drop(['first_name','last_name','email','address','AGT_ID','INT_ID','Prev_ID'],axis=1,inplace=True)

#Resetting index to Loanapp_ID
df.set_index('Loanapp_ID', inplace=True)
df_1.set_index('Loanapp_ID', inplace=True)
df.describe()
df.info()

#Handling numerical missing data
from sklearn .impute import SimpleImputer
imputer1 = SimpleImputer(missing_values=np.nan,strategy="median")
imputer1.fit(df.iloc[:,7:9])
df.iloc[:,7:9]=imputer1.transform(df.iloc[:,7:9])

#Handling categorical missing data
df['Sex'].value_counts()
df['Marital_Status'].value_counts()

imputer2 = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imputer2.fit(df.iloc[:,0:5])
df.iloc[:,0:5]=imputer2.transform(df.iloc[:,0:5])

df['Credit_His'].value_counts()
imputer2.fit(df.iloc[:,9:10])
df.iloc[:,9:10]=imputer2.transform(df.iloc[:,9:10])

#Merging the two incomes into a single one
df["App_Income_1"]=df["App_Income_1"]+df["App_Income_2"]
df.drop("App_Income_2",axis=1,inplace=True)
df.rename(columns= {'App_Income_1':'Total_Income'},inplace=True)

#Checking Distribution of Numeric values

sns.distplot(df["Total_Income"])
sns.boxplot(y="Total_Income",x= "Qual_var",data=df)

sns.distplot(df["CPL_Amount"])
sns.boxplot(y="CPL_Amount",data=df)
sns.boxplot(y="CPL_Amount",x= "Sex",data=df)
sns.boxplot(y="CPL_Amount",x= "Prop_Area",data=df)

#Pure insigts from the data
sns.countplot(x="CPL_Status",data=df,hue="Sex")
sns.countplot(x="CPL_Status",data=df,hue="Marital_Status")
sns.countplot(x="CPL_Status",data=df,hue="Credit_His")
sns.countplot(x="CPL_Status",data=df,hue="Qual_var")
sns.countplot(x="CPL_Status",data=df,hue="SE")
sns.countplot(x="CPL_Status",data=df,hue="Prop_Area")
sns.countplot(x="CPL_Status",data=df,hue="Dependents")
sns.countplot(x="CPL_Status",data=df,hue="CPL_Term")


#Label encoding no of dependents and prop Area
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 

df['SE']= le.fit_transform(df['SE']) #Y:1 N:0
df["Marital_Status"]= le.fit_transform(df["Marital_Status"]) #Y:1   N:0
df['Sex']= le.fit_transform(df['Sex']) #M:1 F:0
df['Dependents']= le.fit_transform(df['Dependents'])
df['Prop_Area']= le.fit_transform(df['Prop_Area']) #Urban:2 Rural:0 Semi-Urban:1
df["Qual_var"]=le.fit_transform(df["Qual_var"])  #Grad:0  NonGrad:1
df['CPL_Status']= le.fit_transform(df['CPL_Status'])#yes:1 No:0

#Data visualization
correlation=df.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)
sns.pairplot(df)

df.drop("SE",axis=1,inplace=True)

#Avoiding the famous dummy variable trap
df=pd.get_dummies(df,columns=["Dependents","Prop_Area"],drop_first=True)

#Splitting into train and validation here.
from sklearn.model_selection import train_test_split
X = df.iloc[:, df.columns != 'CPL_Status'].values
y = df.iloc[:,7].values
X_act=df_1.iloc[:,].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)

#Applying feature scaling to CPL_Amount,CPL_Term,Total_Income in training and validation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train[:,[3,4,5]] = sc_X.fit_transform(X_train[:,[3,4,5]])
X_test[:,[3,4,5]] = sc_X.transform(X_test[:,[3,4,5]])

##Applying feature scaling to CPL_Amount,CPL_Term,Total_Income in real test data
X_act[:,[3,4,5]]=sc_X.transform(X_act[:,[3,4,5]])




# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', max_depth=40,min_samples_split=40,random_state = 0)
classifierDT.fit(X_train, y_train)

# Predicting the Test set results
y_predDT = classifierDT.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predDT)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predDT))
print('Report :')
print(classification_report(y_test, y_predDT))

#Prediction of the actual testset:
y_actDT = classifierDT.predict(X_act)
df_1["DT_OUT"]=y_actDT




# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 100,max_depth=30,min_samples_split=20, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)
# Predicting the Test set results
y_predRF = classifierRF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predRF)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predRF))
print('Report :')
print(classification_report(y_test, y_predRF))

#Prediction of the actual testset:
y_actRF = classifierRF.predict(X_act)
df_1["RF_OUT"]=y_actRF



# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'rbf',random_state = 0)
classifierSVM.fit(X_train, y_train)

# Predicting the Test set results
y_predSVM = classifierSVM.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predSVM)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predSVM))
print('Report :')
print(classification_report(y_test, y_predSVM))

#Prediction of the actual testset:
y_actSVM = classifierSVM.predict(X_act)
df_1["RF_OUT"]=y_actSVM



#Using KNN:
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',algorithm="auto", p = 2)
classifierKNN.fit(X_train, y_train)

y_predKNN = classifierKNN.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predKNN)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predKNN))
print('Report :')
print(classification_report(y_test, y_predKNN))

#Prediction of the actual testset:
y_actKNN = classifierKNN.predict(X_act)
df_1["KNN_OUT"]=y_actKNN



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)

# Predicting the Test set results
y_predLR = classifierLR.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predLR)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predLR))
print('Report :')
print(classification_report(y_test, y_predLR))

#Prediction of the actual testset:
y_actLR = classifierLR.predict(X_act)
df_1["LR_OUT"]=y_actLR



# Importing the MLPClassifier libraries and packages
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(6,6), activation='relu', solver='adam', max_iter=500,random_state=1)
mlp.fit(X_train,y_train)

# Predicting the Test set results
y_predMLP = mlp.predict(X_test)
y_predMLP = (y_predMLP > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_predMLP)
print("Confusion matrix")
print(cm)
print('Accuracy Score :',accuracy_score(y_test,y_predMLP))
print('Report :')
print(classification_report(y_test, y_predMLP))

#Prediction of the actual testset:
y_actMLP = mlp.predict(X_act)
df_1["MLP_OUT"]=y_actMLP

df_1.to_csv('Final Predicted outputs.csv')

