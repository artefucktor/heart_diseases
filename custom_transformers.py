#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:56:52 2023

@author: artefucktor
"""
import bisect
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


######################################################
#   дополнительная колонка – объединение давления  
######################################################

def add_ap(X_):
    X_ = X_.copy()
    X_['ap'] = X_['ap_hi'] + X_['ap_lo']
    return X_


######################################################
#   пересчет возраста в годы  
######################################################

def age_to_year(X_):
    X_ = X_.copy()
    X_['age'] = (X_['age']/365)
    return X_


######################################################
#   коррекция артериального давления  
######################################################

class APTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.ap_mean = None
        self.weight_bucket = [0,60,70,80,90,100]

    def fit(self, X, y = None):
        X_ = X.copy()
                
        # давление лучше всего коррелирует с массой и холестерином
        # группируем по холестерину и массе
        # считаем среднее – из него потом заполним трансформ

        X_['weight_bucket'] = X_['weight'].map(lambda x: bisect.bisect(self.weight_bucket, x))
        self.ap_mean = X_.groupby(['cholesterol','weight_bucket'],as_index=False)[['ap_hi','ap_lo']]\
                         .mean().rename(columns={'ap_hi':'ap_hi_mean','ap_lo':'ap_lo_mean'})
        
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy()
        X_[['ap_lo','ap_hi']] = X_[['ap_lo','ap_hi']].astype('float')
    
        # correct AP hi
        # верхнее давление – убираем отрицательные значения 
        # доводим до приемлемого диапазона – единицы умножаем, а многознаки делим на 10

        X_['ap_hi'] = abs(X_['ap_hi'])

        ap_hi_min,ap_hi_max = 60,250

        mask_min = (X_['ap_hi'] >0 ) & (X_['ap_hi'] < ap_hi_min)
        mask_max = X_['ap_hi'] > ap_hi_max

        while not X_[ mask_min ].empty:
            X_.loc[ mask_min , 'ap_hi'] = X_['ap_hi'] * 10
            mask_min = (X_['ap_hi'] >0 ) & (X_['ap_hi'] < ap_hi_min)

        while not X_[ mask_max ].empty:
            X_.loc[ mask_max , 'ap_hi'] = X_['ap_hi'] / 10
            mask_max = X_['ap_hi'] > ap_hi_max

        # correct AP lo
        # нижнее давление – убираем отрицательные значения 
        # доводим до приемлемого диапазона

        X_['ap_lo'] = abs(X_['ap_lo'])

        ap_lo_min,ap_lo_max = 40,200

        mask_min = (X_['ap_lo'] >0 ) & (X_['ap_lo'] < ap_lo_min)
        mask_max = X_['ap_lo'] > ap_lo_max

        while not X_[ mask_min ].empty:
            X_.loc[ mask_min , 'ap_lo'] = X_['ap_lo'] * 10
            mask_min = (X_['ap_lo'] >0 ) & (X_['ap_lo'] < ap_lo_min)

        while not X_[ mask_max ].empty:
            X_.loc[ mask_max , 'ap_lo'] = X_['ap_lo'] / 10
            mask_max = X_['ap_lo'] > ap_lo_max

        # меняем местами значения давления, если нижнее больше верхнего
        
        mask_to_swap = X_['ap_hi'] < X_.ap_lo
        X_.loc[mask_to_swap,['ap_hi','ap_lo']] = X_.loc[mask_to_swap,['ap_lo','ap_hi']].values

        # если нижнее давление неизвестно, а верхнее меньше 100,
        # то убираем и потом заменим на среднее из фита

        X_.loc[(X_['ap_hi'] < 100) & (X_['ap_lo'] == 0),'ap_hi'] = np.nan
        X_.loc[(X_['ap_lo'] > 100) & (X_['ap_hi'] == 0),'ap_lo'] = np.nan

        # преобразуем нули в нан, чтобы обойтись без фильтров, одни fillna 

        X_.loc[~(X_['ap_hi'].between(60,251)),'ap_hi'] = np.nan
        X_.loc[~(X_['ap_lo'].between(40,201)),'ap_lo'] = np.nan

        # пропуски заполним средними из фита
        
        X_['weight_bucket'] = X_['weight'].map(lambda x: bisect.bisect(self.weight_bucket, x))
        idx = X_.index
        X_ = pd.merge(left=X_,right=self.ap_mean,on=['cholesterol','weight_bucket'],how='left').set_index(idx)     
        X_.loc[X_['ap_hi'].isna(),'ap_hi'] = X_['ap_hi_mean']
        X_.loc[X_['ap_lo'].isna(),'ap_lo'] = X_['ap_lo_mean']

        return X_
    

######################################################
#   коррекция косяков в росте/массе, добавление ИМТ  
######################################################

class BMITransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.weight_mean = None
        self.height_bucket = [0,150,160,170,180]

    def fit(self, X, y = None):
        X_ = X.copy()
                
        # считаем среднюю массу по группам пол-рост
        
        X_['height_bucket'] = X_['height'].map(lambda x: bisect.bisect(self.height_bucket, x))
        self.weight_mean = X_.groupby(['gender','height_bucket'],as_index=False)['weight']\
                             .mean().rename(columns={'weight':'weight_mean'})
        
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy()
            
        def bmi(w, h):
            return (w/((h/100)**2)).round().astype('int')
        
        # добавляем индекс массы тела
        
        X_['bmi'] = bmi(X_['weight'],X_['height'])

        # похоже на ошибку при сборе данных, исправляем
        
        X_.loc[X_['height'] <=100,'height'] = X_['height'] + 100
        X_.loc[X_['height'] >=250,'height'] = X_['height'] - 100

        # если получился неадекватный ИМТ или изначально масса указана неправдоподобная
        # сбрасываем значения в nan 
        
        X_.loc[X_['weight'] <40,'weight'] = np.nan
        X_.loc[X_['bmi']    <15,'weight'] = np.nan

        # заменяем с учетом пола и роста
        
        X_['height_bucket'] = X_['height'].map(lambda x: bisect.bisect(self.height_bucket, x))
        idx = X_.index
        X_ = pd.merge(left=X_,right=self.weight_mean,on=['gender','height_bucket'],how='left').set_index(idx)
        X_.loc[X_['weight'].isna(),'weight'] = X_['weight_mean']
        
        # обновляем ИМТ после всех коррекций
        
        X_['bmi'] = bmi(X_['weight'],X_['height'])
#         X_.drop(['height_bucket','weight_mean'])

        return X_