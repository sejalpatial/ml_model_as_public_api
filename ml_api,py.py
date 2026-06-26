# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:36:06 2026

@author: Acer
"""

from fastapi import FastAPI  #used for creating apis to connect your frontend with model
#FastAPI is a framework used to create APIs. It allows other applications, websites, or mobile apps to send data to our model and receive predictions.
from pydantic import BaseModel #setup the format in whcih input data will be posted to api
import pickle #load our saved model
import json #to convert json to dict

#json - standard format by which data is exhanged between applications and apis
app=FastAPI()

class model_input(BaseModel):
    # for api to know what is the format of data we are giving
    Pregnancies :  int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float 
    DiabetesPedigreeFunction : float
    Age : int

#loading the saved model
diabetes_model = pickle.load(open(r"D:\ml\ml_model_as_Api\python code\trained_model.sav",'rb'))


#creating api

@app.post('/diabetes_prediction') #endpoint - /diabetes_prediction  and post is used to send data to server/api
#input is in form of json , therefore we convert it into dict
def diabetes_pred(input_parameters: model_input): 
    
    input_dictionary = input_parameters.model_dump()
    
    # all the words inside ' ' are keys and we are accesiing the values of keys  in variable
    preg =input_dictionary['Pregnancies']
    glu =input_dictionary['Glucose']
    bp =input_dictionary['BloodPressure']
    skin=input_dictionary['SkinThickness']
    insulin =input_dictionary['Insulin']
    bmi =input_dictionary['BMI']
    dpf =input_dictionary['DiabetesPedigreeFunction']
    age =input_dictionary['Age']
    
    # all variables are stored in list
    input_list=[preg,glu,bp,skin,insulin,bmi,dpf,age]
    
    prediction = diabetes_model.predict([input_list])

    if prediction[0]==0:
        return {"PREDICTION":'person is not diabetic'}
    else:
        return {'PREDICTION':" person is diabetic"}
    
# Client File
#      |
#      |  Sends input data as JSON
#      ↓
# FastAPI Endpoint (/diabetes_prediction)
#      |
#      |  Receives request and validates input
#      ↓
# Pydantic Model (BaseModel)
#      |
#      |  Converts JSON into Python object/dictionary
#      ↓
# Creates Input List
#      |
#      |  Arranges values in the same order as used during model training
#      ↓
# Saved ML Model (.sav)
#      |
#      |  Makes prediction using model.predict()
#      ↓
# FastAPI
#      |
#      |  Returns prediction to the client
#      ↓
# Client File
#      |
#      |  Receives response and prints/displays the result
#      ↓
# Output:
# Person is Diabetic
# (or)
# Person is Not Diabetic












