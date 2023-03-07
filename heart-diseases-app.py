#!/usr/bin/env python
# coding: utf-8

#%% LOAD LIBRARIES

import streamlit as st
import pandas as pd
import time
import os

from joblib import load

#%% LOAD MODEL

model      = load(os.path.dirname(__file__) + '/XGB_classifier.joblib')
preprocess = load(os.path.dirname(__file__) + '/scaler_encoder.joblib')

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

extreme_warning = 'При экстремальных параметрах оценка риска может быть неточной'

gender_select = st.radio('Пол',('Мужчина','Женщина'),label_visibility='hidden')

age_year = st.slider('Возраст', min_value=25, max_value=65)
height = st.slider('Рост', min_value=120, max_value=220)
if height >200:
    st.info(extreme_warning, icon="⚠️")
weight = st.slider('Масса', min_value=40, max_value=170)
if weight >200:
    st.info(extreme_warning, icon="⚠️")
if weight >height:
    st.error('Предупреждение: очень большая масса для указанного роста', icon="⚠️")

st.subheader('Давление')
ap1,ap2 = st.columns(2,gap='large')

with ap1:
    ap_hi = st.slider('систолическое', min_value=60, max_value=240)
with ap2:
    ap_lo = st.slider('диастолическое', min_value=40, max_value=200)
if ap_lo >= ap_hi:
    st.error('Предупреждение: Систолическое давление ниже диастолического', icon="⚠️")
if (ap_lo >180) or (ap_hi >200):
    st.info(extreme_warning, icon="⚠️")

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
    smoke_select = st.radio('Курение',('Курит','Не курит'))
with col3:
    alco_select = st.radio('Алкоголь',('употребляет','не употребляет'))

#%% CALCULATIONS

# prepare data to model

age    = age_year * 365

gender = 2 if gender_select=='Мужчина' else 1
smoke  = 0 if smoke_select=='Не курит' else 1
alco   = 0 if alco_select=='не употребляет' else 1
active = 0 if active_select=='малоподвижный' else 1

bmi    = weight/(height**2)*10000

weight_bucket = {
                  weight < 60 : 1,
            60 <= weight < 70 : 2,
            70 <= weight < 80 : 3,
            80 <= weight < 90 : 4,
                  weight >=90 : 5
                  }[True]

test_feature   = ap_hi * ap_lo
test_feature_2 = gluc * cholesterol

risk_data = pd.DataFrame({'age'           : [age],
                          'gender'        : [gender],
                          'height'        : [height],
                          'bmi'           : [bmi],
                          'weight_bucket' : [weight_bucket],
                          'ap_hi'         : [ap_hi],
                          'ap_lo'         : [ap_lo],
                          'cholesterol'   : [cholesterol],
                          'gluc'          : [gluc],
                          'smoke'         : [smoke],
                          'alco'          : [alco],
                          'active'        : [active],
                          'test_feature'  : test_feature,
                          'test_feature_2': test_feature_2,
                       })

# predict proba
risk = model.predict_proba(preprocess.transform(risk_data))[:,1][0]
risk_percent = risk * 100

#%% FEATURE IMPORTANCES

feature_names = preprocess.get_feature_names_out()
feature_importances = pd.DataFrame({'features':feature_names,
                                    'feature_importances':model.feature_importances_})

# SUM importances AP_HI+AP_LO to single variable AP
# (optimize the number of output variables)

feature_importances.loc[len(feature_importances)] = {'features':'sum_ap',
                                                     'feature_importances':feature_importances[feature_importances['features']\
                                                                                               .isin(['num__ap_hi',
                                                                                                      'num__ap_lo'])]\
                                                         ['feature_importances'].sum()}
feature_importances.loc[len(feature_importances)] = {'features':'sum_weight',
                                                     'feature_importances':feature_importances[feature_importances['features']\
                                                                                               .isin(['num__bmi',
                                                                                                      'num__weight_bucket'])]\
                                                         ['feature_importances'].sum()}

# select and prepare features to output
# выводим только переменные параметры, поддающиеся коррекции
# (вряд ли пациент может произвольно менять рост, пол, возраст)

features_to_output = feature_importances[feature_importances['features'].isin(['sum_ap',
                                                                               'sum_weight',
                                                                               'pss__cholesterol',
                                                                               'pss__gluc',
                                                                               'pss__smoke',
                                                                               'pss__alco',
                                                                               'pss__active'
                                                                               ])].copy()
to_replace  = {'sum_weight'      :'Лишний вес',
               'sum_ap'          :'Давление',
               'pss__cholesterol':'Холестерин',
               'pss__gluc'       :'Глюкоза',
               'pss__smoke'      :'Курение',
               'pss__alco'       :'Алкоголь',
               'pss__active'     :'Образ жизни'}

features_to_output['features'] = features_to_output['features'].replace(to_replace)
features_to_output['feature_importances'] = features_to_output['feature_importances'] * risk_percent
features_to_output = features_to_output.sort_values(by='feature_importances', ascending=False)

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
        
    
    
    





    
