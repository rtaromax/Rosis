# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:25:21 2016

@author: rtaromax
"""

import pandas as pd
import random
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt


df_visit1 = pd.read_csv('/Users/rtaromax/Documents/Rosis/data/visit data/visit1.csv')
df_visit2 = pd.read_csv('/Users/rtaromax/Documents/Rosis/data/visit data/visit2.csv')
df_protein = pd.read_csv('/Users/rtaromax/Documents/Rosis/data/protein/protein_data.csv')
df_smc = pd.read_csv('/Users/rtaromax/Documents/Rosis/data/metabolic/SMC_P0204_GCMS.csv')

target_list = ['STUDIELOPNUMMER','BHBA1CSF','SINSSF','PGLUCSF','GLP1','GLUCAGON']


def protein_visit(df_protein, visit):
    df_protein_visit = df_protein[df_protein['Visitno']==visit]
    df_protein_visit = df_protein_visit.set_index(['NPX'])
    del df_protein_visit['Patid']
    del df_protein_visit['Visitno']
    df_protein_visit = df_protein_visit.convert_objects(convert_numeric=True)
    
    return df_protein_visit
    

def smc_preprocessing(df_smc):
    del df_smc['Visit']
    del df_smc['PatientID']
    df_smc = df_smc.set_index(['Patient ID'])
    df_smc.replace(to_replace=0, value=np.nan, inplace=True)
    df_smc = df_smc.fillna(df_smc.mean())/100000
    
    return df_smc
    
    
def visit_target(df_visit, target_list):
    df_target_visit = df_visit[target_list]
    df_target_visit['STUDIELOPNUMMER'] = df_target_visit['STUDIELOPNUMMER'].str.replace('da', '166101')
    df_target_visit = df_target_visit.set_index(['STUDIELOPNUMMER'])
    
    return df_target_visit
    

#protein data 1
df_protein_1 = protein_visit(df_protein, 1)
df_protein_correlation = df_protein_1.corr()

#protein data 2
df_protein_2 = protein_visit(df_protein, 2)

#SMC
df_smc = smc_preprocessing(df_smc)

#visit1
df_target_1 = visit_target(df_visit1, target_list)

#visit2
df_target_2 = visit_target(df_visit2, target_list)


#df_protein_1_imp = df_protein_1.fillna(df_protein_1.mean())
df_target = pd.concat([df_target_1, df_target_2])
df_protein = pd.concat([df_protein_1, df_protein_2])
df_merge = pd.DataFrame(df_target['BHBA1C']).merge(df_protein, right_index = True, left_index = True, how = 'inner')
df_merge_n = pd.concat([df_merge, df_smc.ix[:,0:]], axis=1)
df_merge_n = df_merge_n.dropna()



rows = random.sample(list(df_merge_n.index), int(0.8*len(df_merge_n.index)))
df_train_70p = df_merge_n.loc[rows]
df_test = df_merge_n.drop(rows)

'''
agglo = cluster.FeatureAgglomeration(n_clusters=20)
agglo.fit(df_train_70p.ix[:,1:])
df_train_reduced = agglo.transform(df_train_70p.ix[:,1:])
agglo.fit(df_test.ix[:,1:])
df_test_reduced = agglo.transform(df_train_70p.ix[:,1:])
'''


#randomforest
clf = RandomForestRegressor(n_jobs=2, n_estimators=100)

r_sum = 0
mape_sum = 0
for i in range(20):
    clf.fit(df_train_70p.ix[:,1:], df_train_70p.ix[:,0])
    importances = clf.feature_importances_

    y_pred = clf.predict(df_test.ix[:,1:])
    y_true = df_test.ix[:,0]
    results = pd.concat([pd.DataFrame(y_pred), y_true.reset_index()], axis=1)

    mse = mean_squared_error(y_true.tolist(), y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred,y_true)
    r_sum = r_sum + r_value
    mape_sum = mape_sum + mape

print(r_sum/20)
print(mape_sum/20)

#plot
plt.hist(df_merge['GLP1'].dropna())
plt.title("GLP1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
