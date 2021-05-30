#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 20:08:58 2021

@author: vimal
"""
import pickle
import numpy as np
import streamlit as st 

pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)

pickle_in = open("Featurescale.pkl","rb")
sc=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def prediction(sex,marital_status,dependents,qualification,proparea,total_income,cpl_amount,cpl_term,credit_history):
    if sex=="Male":
       sex=1
    else:
       sex=0
       
    if marital_status=="Unmarried":
       marital_status=0;
    else:
       marital_status=1;
       
    if qualification=="Graduated":
       qualification=0;
    else:
       qualification=1;
       
    if dependents=="0":
       dependent1=0
       dependent2=0
       dependent3=0
    elif dependents=="1":
       dependent1=1
       dependent2=0
       dependent3=0
    elif dependents=="2":
       dependent1=0
       dependent2=1
       dependent3=0
    else:
       dependent1=0
       dependent2=0
       dependent3=1
       
    if proparea=="Urban":
       proparea1=0
       proparea2=1
    elif proparea=="Rural":
       proparea1=0
       proparea2=0
    else:
       proparea1=1
       proparea2=0
       
    if credit_history=="Unclear Debts":
        credit_history=0
    else:
        credit_history=1
        
    cpl_amount=cpl_amount/1000
    arr=np.array([[sex,marital_status,qualification,total_income,cpl_amount,cpl_term,credit_history,dependent1,dependent2,dependent3,proparea1,proparea2]])
    arr[:,[3,4,5]]=sc.transform(arr[:,[3,4,5]])
    prediction=model.predict(arr);
    
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred

def main():
      st.title("Consumer Loan Approval Prediction ")
      html_temp = """
      <div style="background-color:tomato;padding:10px">
      <h2 style="color:white;text-align:center;">Streamlit Loan Authenticator ML App </h2>
      </div>
      """
      # display the front end aspect
      st.markdown(html_temp,unsafe_allow_html=True)
      
      # following lines create boxes in which user can enter data required to make prediction 
      sex = st.selectbox('Gender',("Male","Female"))
      
      marital_status = st.selectbox('Marital Status',("Unmarried","Married"))
      
      dependents = st.selectbox('Dependents',("0","1","2","3+"))
      
      qualification = st.selectbox('Qualification',("Graduated","Not Graduated"))
      
      proparea =st.selectbox('Area',("Urban","Rural","Semi-Urban"))
      
      total_income = st.number_input("Applicants Monthly Income",min_value=0,step=1000);
      
      cpl_amount = st.number_input("Total Loan Amount",min_value=0,step=1000)
      
      cpl_term= st.selectbox('Loan term',(120,180,240,360,480))
      
      credit_history=st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
      result=""
      
      # when 'Predict' is clicked, make the prediction and store it 
      if st.button("Predict"): 
        result = prediction(sex,marital_status,dependents,qualification,proparea,total_income,cpl_amount,cpl_term,credit_history) 
        st.success('Your loan is {}'.format(result))
        print(cpl_amount)
      
if __name__=='__main__':
    main()


