#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:29:24 2021

@author: vimal
"""

#Importing the library
import numpy as np
import pandas as pd
import pickle

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
df['CPL_Term'].describe()

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
df['Total_Income'].describe()


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

#Saving the transform object in pickle
pickle_out = open("Featurescale.pkl", mode = "wb") 
pickle.dump(sc_X, pickle_out) 
pickle_out.close()


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
#y_actRF = classifierRF.predict(X_act)
#df_1["RF_OUT"]=y_actRF

# saving the model 
pickle_out = open("model.pkl", mode = "wb") 
pickle.dump(classifierRF, pickle_out) 
pickle_out.close()


