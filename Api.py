# Importation des librairies
#pip install uvicorn[all]
#pip install fastapi[all]
#pip install fastapi_csv
# import model 
import sys
sys.path.insert(0, '../src')
import uvicorn
from fastapi import FastAPI 
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
from lightgbm import LGBMClassifier
import shap 
import plotly.graph_objects as go
from enum import Enum

# Création de l'instance API et chargement du dataset
app = FastAPI()


# Prérequis API (data + model + fonction de prédiction)
data = pd.read_csv("relevant_data_50_for_dashboard.csv", index_col='SK_ID_CURR', encoding ='utf-8')

def load_model():
        '''loading the trained model'''
        pickle_in = open('classifier_lgbm_model_with_relevantdata50.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf
    
clf = load_model()

# Récupération de la prédiction du crédit pour les clients 
def load_prediction(data, id, clf):
    X=data.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score


@app.get('/')
def home():
    return {'text': 'Welcome to Home Credit Prediction'}

# Endpoint qui retourne le score en fonction de l'id client
@app.get('/predict')
async def read_items(id: int):
    if id < 100002 or id > 123325:
        return 'id client inconnu'
    score = round(float(load_prediction(data, float(id), clf)), 2)
    return {'score': score}

# Exécuter l'API avec uvicorn
#    Exécution sur http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn app:app --reload 