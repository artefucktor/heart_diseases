#!/usr/bin/env python
# coding: utf-8

#%% LOAD LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import time
import os,sys

from joblib import load

#%% CUSTOM TRANSFORMERS USED BY MODEL

sys.path.append(os.path.dirname(__file__))
from custom_transformers import age_to_year,BMITransformer,APTransformer

#%% LOAD MODEL

model = load(os.path.dirname(__file__) + '/model.joblib')

#%% HEADER

st.set_page_config(page_title='Здоровое сердце', 
                   page_icon='❤️', 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

with st.container():
    st.header('Советник ❤️ кардиолога')
    st.markdown('*Вероятность развития сердечно-сосудистых заболеваний*')

#%% INPUT FORMS

gender_select = st.radio('Пол',('Мужчина','Женщина'),label_visibility='hidden')

age_year = st.slider('Возраст', min_value=30, max_value=100)
height = st.slider('Рост', min_value=100, max_value=220)
weight = st.slider('Вес', min_value=40, max_value=200)

st.subheader('Давление')
ap1,ap2 = st.columns(2,gap='large')
with ap1:
    ap_hi = st.slider('верхнее', min_value=60, max_value=220)
with ap2:
    ap_lo = st.slider('нижнее', min_value=40, max_value=200)

st.subheader('Биохимия')
bl1,bl2 = st.columns(2,gap='large')
with bl1:
    cholesterol = st.select_slider('Холестерин',(1,2,3))
with bl2:
    gluc = st.select_slider('Глюкоза',(1,2,3))

st.subheader('Привычки')
col1,col2,col3 = st.columns(3,gap='large')
with col1:
    active_select = st.radio('Образ жизни',('активный','малоподвижный'))
with col2:
    smoke_select = st.radio('Курение',('Курю','Не курю'))
with col3:
    alco_select = st.radio('Алкоголь',('употребляю','не употребляю'))

#%% CALCULATIONS

# prepare data to model

gender = 2 if gender_select == 'Мужчина' else 1
age = age_year *365
smoke = 0 if smoke_select=='Не курю' else 1
alco = 0 if alco_select=='не употребляю' else 1
active = 0 if active_select=='малоподвижный' else 1

risk_data = pd.DataFrame({'age'   : [age],
                          'gender': [gender],
                          'height': [height],
                          'weight': [weight],
                          'ap_hi' : [ap_hi],
                          'ap_lo' : [ap_lo],
                          'cholesterol': [cholesterol],
                          'gluc'  : [gluc],
                          'smoke' : [smoke],
                          'alco'  : [alco],
                          'active': [active]
                        })

# predict proba

risk = model.predict_proba(risk_data)[:,1][0]
risk_percent = risk * 100

#%% FEATURE IMPORTANCES

to_scale  = ['age','weight','bmi','ap_hi','ap_lo','cholesterol','gluc']
to_encode = ['gender','smoke','alco','active']
cat_names = model['pre'].steps[1][1].named_transformers_['cat'].get_feature_names_out()
num_names = model['pre'].steps[1][1].named_transformers_['num'].get_feature_names_out(to_scale)
all_names = np.concatenate([cat_names, num_names])
feature_importances = pd.DataFrame({'features':all_names,
                                    'feature_importances':model._final_estimator.feature_importances_})

# SUM importances WEIGHT+BMI and AP_HI+AP_LO to common variables weight and ap
# (optimize the number of output variables)

feature_importances.loc[len(feature_importances)] = {'features':'common_weight',
                                                     'feature_importances':feature_importances[feature_importances['features'].isin(['weight','bmi'])]['feature_importances'].sum()}
feature_importances.loc[len(feature_importances)] = {'features':'common_ap',
                                                     'feature_importances':feature_importances[feature_importances['features'].isin(['ap_hi','ap_lo'])]['feature_importances'].sum()}

# select and prepare features to output

features_to_output = feature_importances[feature_importances['features'].isin(['common_weight',
                                                                               'common_ap',
                                                                               'cholesterol',
                                                                               'gluc',
                                                                               'smoke_1',
                                                                               'alco_1',
                                                                               'active_1'
                                                                               ])].copy()
to_replace  = {'common_weight':'Лишний вес',
               'common_ap':'Давление',
               'cholesterol':'Холестерин',
               'gluc':'Глюкоза',
               'smoke_1':'Курение',
               'alco_1':'Алкоголь',
               'active_1':'Образ жизни'}
features_to_output['features'] = features_to_output['features'].replace(to_replace)
features_to_output['feature_importances'] = features_to_output['feature_importances'] * risk_percent
features_to_output = features_to_output.sort_values(by='feature_importances',ascending=False)

#%% RESULT OUTPUT

# colorize progress bar

st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, yellow , orange);
        }
    </style>
""", unsafe_allow_html=True)

# output risk

st.markdown('---')
st.subheader('Вероятность ССЗ')

level_bar = st.progress(0)
for i in range(round(risk_percent)):
    time.sleep(0.001)
    level_bar.progress(i)
    
level_risk = {
          risk_percent < 25 : 'Низкая',
    25 <= risk_percent < 50 : 'Умеренная',
    50 <= risk_percent < 75 : 'Высокая',
    75 <= risk_percent      : 'Очень высокая'
}[True]

st.subheader(str(level_risk) + ' ' + str(round(risk_percent)) + '%') 

# output factors

st.markdown('---')
st.subheader('Основные факторы риска')

for index, row in features_to_output.iterrows():
    feat_level = round(row['feature_importances'])
    if feat_level >0:
        feat_bar = st.progress(feat_level,row['features'])
        
    
    
    





    