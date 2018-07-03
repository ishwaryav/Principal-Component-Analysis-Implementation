#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:23:51 2018

@author: ishwaryavaradarajan
"""

import pandas as pd
import numpy as np
train = pd.read_csv('train.dat', header=None, sep=' ', quoting=3)
label = pd.read_csv('train.labels', header=None, sep=' ', quoting=3)
test = pd.read_csv('test.dat', header=None, sep=' ', quoting=3)

X_train = train.iloc[:,0:887]
y_train = label.iloc[:,0]
X_test = test.iloc[:,0:887]

#training
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)


#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

X_train_mean = np.mean(X_train, axis=0)
X_train = X_train - X_train_mean

X_test_sm = np.mean(X_test, axis=0)
X_test = X_test - X_test_sm

from sklearn.decomposition import PCA
pca = PCA(n_components = 28)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#model = LDA(n_components = 20)
#X_train = model.fit_transform(X_train, y_train)
#X_test = model.transform(X_test)

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)

from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.3, n_estimators=2000, max_depth=7, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
xgb.fit(X_train, y_train)
y_pred=xgb.predict(X_test)

#78.2 for n_components = 200 and n_estimators=500 in clp with LDA
#80.53 for n_components = 20 and n_estimators=100 in clp with PCA
#80.7 for n_components = 20 and n_estimators=200 in clp with PCA
#81.84 for n_components = 20 and n_estimators=200 in clp with PCA, RF, centering and no std scaling
#learning_rate=0.1, n_estimators=1000, max_depth=7, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27) - 82.1%
#learning_rate=0.3 and above parameters give 82.3
#learning_rate=0.3, n_estimators=2000 give 82.4

#from sklearn.metrics import f1_score
#score = f1_score(y_test, y_pred,average='weighted')
#print (score)

output = pd.DataFrame(data=y_pred)
output.to_csv("output.dat", index=False, quoting=3, header=None)

#with open('train.dat') as f1:
#    next(f1)
#    df1=pd.DataFrame(l.strip().split() for l in f1)

#with open('train.labels') as f2:
#    next(f2)
#    df2=pd.DataFrame(l.strip().split() for l in f2)
    
#with open('test.dat') as f3:
#    next(f3)
#    df3=pd.DataFrame(l.strip().split() for l in f3)