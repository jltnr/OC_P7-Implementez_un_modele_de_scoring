### Dashboard with Streamlit 

#Importation des librairies
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
import pickle
from lightgbm import LGBMClassifier
import shap 
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
from enum import Enum

######
##### Test de code interactif ####

# import streamlit as st 
# import plotly.express as px
# import pandas as pd


# data = {
#     'City': ['c1', 'c2', 'c3', 'c1', 'c3', 'c2', 'c1'],
#     'product_line': ['p1', 'p2', 'p3', 'p3', 'p2', 'p1', 'p4'],
#     'quantity': [8, 4, 3, 12, 5, 6, 4],
#     'gross_income': [250, 150, 300, 250, 300, 400, 500],
#     'gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male']
# }

# df = pd.DataFrame(data)
# st.write(df)


# with st.expander('Favorite product by Gender within city'):
#     column1, column2 = st.columns([3,1])
        
#     # Allow the user to select a gender.
#     selected_gender = st.radio('What is your Gender:', df.gender.unique(), index = 0)

#     # Apply gender filter.
#     gender_product = df[df['gender'] == selected_gender]

#     # Allow the user to select a city.
#     select_city = column2.selectbox('Select City', df.sort_values('City').City.unique())

#     # Apply city filter
#     city_gender_product = gender_product[gender_product['City'] == select_city]

#     # Use the city_gender_product dataframe as it has filters for gender and city.
#     fig = px.histogram(city_gender_product.sort_values('product_line') ,x='product_line', y='gross_income', color = 'product_line',)

#     if selected_gender == 'Male':
#         st.write('What men buy most!')
#     else:
#         st.write('What female buy most!')

#     st.plotly_chart(fig, use_scontainer_width=True) 

    
#####

#Application du titre principal 
# st.title('Prédiction des crédits de consommation')

#Chargement du dataframe et du modèle
model = pickle.load(open('classifier_lgbm_model.pkl', 'rb'))
data = pd.read_csv('relevant_data_for_dashboard.csv', index_col='SK_ID_CURR', encoding ='utf-8')
data = data.drop(["index"], axis=1)
target = data.iloc[:, -1:]


class Gender(Enum):
    MALE = 0.0
    FEMALE = 1.0

class FamilyStatus(Enum):
    NOT_MARRIED = 0.0
    MARRIED = 1.0 
    

# st.dataframe(data)


def load_model():
        '''loading the trained model'''
        pickle_in = open('classifier_lgbm_model.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


clf = load_model()
    
#Récupération des informations générales clients 
@st.cache
def load_infos_gen(data):
    nb_credits = data.shape[0]
    rev_moy = round(data["AMT_INCOME_TOTAL"].mean(),2)
    credits_moy = round(data["AMT_CREDIT"].mean(), 2)
    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets

#Récupération de l'identifiant client 
def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client

#Récupération de l'âge de la population de l'échantillon 
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/-365), 2)
    return data_age

#Récupération du revenu de la population de l'échantillon 
@st.cache
def load_income_population(data):
    df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

#Récupération du crédit de la population de l'échantillon 
@st.cache
def load_amt_credit_population(data):
    amt_credit_pop = pd.DataFrame(data["AMT_CREDIT"])
    # df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return amt_credit_pop

#Récupération des défauts de paiement de la population de l'échantillon 
@st.cache
def load_obs_defaultpayment_population(data):
    obs_def_payment_pop = pd.DataFrame(data["OBS_30_CNT_SOCIAL_CIRCLE"])
    return obs_def_payment_pop

#Récupération des taux de paiment de la population de l'échantillon 
@st.cache
def load_payment_rate_population(data):
    payment_rate_pop = pd.DataFrame(data["PAYMENT_RATE"])
    return payment_rate_pop

#Récupération de la prédiction du crédit pour les clients 
@st.cache
def load_prediction(data, id, clf):
    X=data.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score


#Chargement de l'identifiant client 
id_client = data.index.values

#Affichage du titre
html_temp = """
<div style="background-color: LightSeaGreen; padding:10px; border-radius:10px">
<h1 style="color: white; font-size: 30px; text-align:center">Dashboard for consumer credit prediction</h1>
</div>
<p style="font-size: 15px; font-weight: bold; text-align:center">Credit decision support…</p>
"""
st.markdown(html_temp, unsafe_allow_html=True)


### Création du menu sur le côté gauche ### 
#Sélection de l'id du client 
st.sidebar.header("**General Information**")

#Chargement de la boîte de sélection du client 
chk_id = st.sidebar.selectbox("Client ID", id_client)

#Chargement des informations générales 
nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

#Nombre total de crédits de l'échantillon 
st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

#Revenu moyen des clients dans l'échantillon 
st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

#Crédit moyen des clients dans l'échantillon 
st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

#PieChart concernant le nombre de crédits acceptés et refusés dans l'échantillon 
#st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(3,3))
plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
plt.title("Customer creditworthiness chart", fontweight = 'bold')
st.sidebar.pyplot(fig)



### Création de la page principale ###
st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Features importance global for consumer credit prediction</h1>", unsafe_allow_html=True)

#Features Importance global 
X = data.iloc[:, :-1]
fig, ax = plt.subplots(figsize=(10, 10))
explainer = shap.TreeExplainer(load_model())
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[0], X, color_bar=False, plot_size=(5, 5))
st.pyplot(fig)



#Affichage des informations clients : Genre, Age, Statut familial, Enfants, …
with st.expander("**Customer information display**"):
    #Affichage de l'identifiant du client sélectionné à partir du menu 
    st.write("Customer ID selection :", chk_id)
    
    if st.checkbox("Show customer information ?"):
        
        infos_client = identite_client(data, chk_id)
        # client_gender = infos_client.iloc[0]['CODE_GENDER']
        # infos_client.iloc[0]['CODE_GENDER'] = Gender(
        client_gender = Gender(infos_client.iloc[0]['CODE_GENDER']).name
        client_age = ((infos_client.iloc[0]['DAYS_BIRTH'] /-365))
        client_status = FamilyStatus(infos_client.iloc[0]['NAME_FAMILY_STATUS_Married']).name
        client_children = infos_client.iloc[0]['CNT_CHILDREN']
        st.write("**Infos clients**", infos_client)
        st.write("**Gender :**", (client_gender))
        st.write("**Age :**", round(client_age))
        st.write("**Family status :**", client_status)
        st.write("**Number of children :**", client_children)

        #Distribution par âge
        # data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        # sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        sns.histplot(((data['DAYS_BIRTH'])/-365), edgecolor = 'k', color="teal", bins=30)
        ax.axvline(int(client_age), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)


        st.subheader("*Income (USD)*")
        client_income = infos_client.iloc[0]['AMT_INCOME_TOTAL']
        client_credit = infos_client.iloc[0]['AMT_CREDIT']
        client_annuity = infos_client.iloc[0]['AMT_ANNUITY']
        client_property_credit = infos_client.iloc[0]['AMT_GOODS_PRICE']
        st.write("**Income total :**", round(client_income))
        # st.write("**Income total :**{:.0f}" .format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Credit amount :**", round(client_credit))
        st.write("**Credit annuities :**", round(client_annuity))
        st.write("**Amount of property for credit :**", round(client_property_credit))

        #Distribution par revenus
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline(int(client_income), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        #Distribution des crédits demandés dans l'échantillon 
        data_amt_credit = load_amt_credit_population(data)
        client_credit = infos_client.iloc[0]['AMT_CREDIT']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_amt_credit["AMT_CREDIT"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline(int(client_credit), color="green", linestyle='--')
        ax.set(title='Customer credit', xlabel='Credit (USD)', ylabel='')
        st.pyplot(fig)

        #Distribution des défauts de paiements sur 30 jours dans l'échantillon 
        data_defpayment = load_obs_defaultpayment_population(data)
        client_defpayment = infos_client.iloc[0]['OBS_30_CNT_SOCIAL_CIRCLE']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_defpayment["OBS_30_CNT_SOCIAL_CIRCLE"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline(int(client_defpayment), color="green", linestyle='--')
        ax.set(title='Customer default payment', xlabel='Number of default payment (on 30 days)', ylabel='')
        st.pyplot(fig)
        
        #Distribution des taux de paiement ("payment_rate") dans l'échantillon 
        data_payment_rate = load_payment_rate_population(data)
        client_payment_rate = infos_client.iloc[0]['PAYMENT_RATE']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_payment_rate["PAYMENT_RATE"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline((client_payment_rate), color="green", linestyle='--')
        ax.set(title='Customer payment rate', xlabel='Payment rate', ylabel='')
        st.pyplot(fig)

        
#Customer probability repayment display
with st.expander("**Credit default probability**"):

    infos_client = identite_client(data, chk_id)
    client_target = infos_client.iloc[0]['TARGET']
    prediction = load_prediction(data, chk_id, clf)
    
    formatted_prediction = round(float(prediction)*100, 2)
    st.write("**Your credit default probability is :** {:.0f} %".format(formatted_prediction))
    
    # if round(float(prediction)*100, 2) == 100 :
    #     st.error('Your credit application has been rejected! Please contact customer support for more information.', icon="❌")
    # else : 
    #     st.success('Congratulations! Your credit application has been accepted!', icon="✅")
    if client_target == 1.0 :
        st.error('Your credit application has been rejected! Please contact customer support for more information.', icon="❌")
    else : 
        st.success('Congratulations! Your credit application has been accepted!', icon="✅")
        
    option = {
        "tooltip": {
            "formatter": '{a} <br/>{b} : {c}%'
        },
        "series": [{
            "name": 'Credit default probability',
            "type": 'gauge',
            "startAngle": 180,
            "endAngle": 0,
            "progress": {
                "show": "true"
            },
            "radius":'100%', 

            "itemStyle": {
                "color": '#5499C7',
                "shadowColor": 'rgba(0,138,255,0.45)',
                "shadowBlur": 10,
                "shadowOffsetX": 2,
                "shadowOffsetY": 2,
                "radius": '55%',
            },
            "progress": {
                "show": "true",
                "roundCap": "true",
                "width": 15
            },
            "pointer": {
                "length": '60%',
                "width": 8,
                "offsetCenter": [0, '5%']
            },
            "detail": {
                "valueAnimation": "true",
                "formatter": '{value}%',
                "backgroundColor": '#5499C7',
                "borderColor": '#999',
                "borderWidth": 4,
                "width": '60%',
                "lineHeight": 20,
                "height": 20,
                "borderRadius": 188,
                "offsetCenter": [0, '40%'],
                "valueAnimation": "true",
            },
            "data": [{
                "value": round(float(prediction)*100, 2),
                "name": 'Credit default probability'
            }]
        }]
    };


    st_echarts(options=option, key="1")

    
#Customer solvability display
with st.expander("**Customer file analysis**"):
    
    #Customer solvability display
    # st.header("**Customer file analysis**")

    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    
    #Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = data.iloc[:, :-1]
        X = X[X.index == chk_id]
        number = st.slider("Pick a number of features…", 0, 20, 5)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
        
        # if st.checkbox("Need help about feature description ?") :
            # list_features = description.index.to_list()
            # feature = st.selectbox('Feature checklist…', list_features)
            # st.table(description.loc[description.index == feature][:1])
        
        # else:
        #     st.markdown("<i>…</i>", unsafe_allow_html=True)

#Création d'un graphique interactif 
# income_options = data['AMT_INCOME_TOTAL'].unique().tolist()
# income_bar = st.selectbox('Which income would you like to see ?', income_options, 0)
# data = data[data['AMT_INCOME_TOTAL'] == income_bar]

# # fig = px.scatter(data, x='TARGET', y='AMT_INCOME_TOTAL', color='TARGET', animation_frame='AMT_INCOME_TOTAL', animation_group='TARGET')
# # fig.update_layout(width=800)
# # st.write(fig)


# number_income = st.slider("Choose an income", 0, 378900)
# fig = px.bar(data, x=, y="AMT_INCOME_TOTAL", color="TARGET", animation_frame="AMT_INCOME_TOTAL", animation_group="TARGET", range_y=[0,4000000000])
# st.write(fig)
    