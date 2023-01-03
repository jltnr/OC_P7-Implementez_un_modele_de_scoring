# 1. Importation des librairies
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

# 2. Création de l'instance API et chargement du dataset
app = FastAPI()
data = pd.read_csv("relevant_data_50_for_dashboard.csv")


@app.get('/')
def home():
    return {'text': 'Welcome to Home Credit Prediction'}

# 3. Exposer la fonctionnalité de prédiction, faire une prédiction à partir des données JSON passées et retourner la valeur prédite
@app.post('/predict')
def predict_target_client(data):
    X = data.drop(['SK_ID_CURR'], axis=1)
    model = pickle.load(open('classifier_lgbm_model_with_relevantdata50.pkl', 'rb'))
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba 

# def predict():
#     model = pickle.load(open("classifier_lgbm_model_with_relevantdata50.pkl", 'rb'))
#     if model:
#         try:
#             json_ = request.json
#             print(json_)
#             query = pd.DataFrame(json_)           
#             y_pred = model.predict(query)
#             y_proba = model.predict_proba(query)
            
#             return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

#         except:

#             return jsonify({'trace': traceback.format_exc()})
#     else:
#         print ('Problem loading the model')
#         return ('No model here to use')

# def predictByClientId():
#     model = pickle.load(open("classifier_lgbm_model_with_relevantdata50.pkl", 'rb'))
#     if model:
#         try:
#             json_ = request.json
#             print(json_)
#             sample_size = 10000
            
#             print(json_)  

#             sample_size= 20000
#             data_set = data = pd.read_csv("relevant_data_50_for_dashboard.csv",nrows=sample_size)
#             client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['SK_ID_CURR','TARGET'],axis=1)
#             print(client)

#             y_pred = model.predict(client)
#             y_proba = model.predict_proba(client)
            
#             return jsonify({'prediction': str(y_pred[0]),'prediction_proba':str(y_proba[0][0])})


#         except:

#             return jsonify({'trace': traceback.format_exc()})
#     else:
#         print ('Problem loading the model')
#         return ('No model here to use')

# 4. Exécuter l'API avec uvicorn
#    Exécution sur http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn app:app --reload 
