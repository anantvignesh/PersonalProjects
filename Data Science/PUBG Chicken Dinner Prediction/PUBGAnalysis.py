# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:21:57 2019

@author: Anant Vignesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#scikit learn library 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestRegressor
#scikit learn library

#-----------------CUSTOM FUNSTIONS------------------------#

#CUSTOM FUNSTIONS
def plot_corr(df):
    corr=df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

#-----------------CUSTOM FUNSTIONS------------------------#


#-----------------READING DATASET------------------------#

dataset = pd.read_csv('E:/MS COMPUTER SCIENCE/MS PROJECTS/OWN PROJECTS/PUBG Chicken Dinner Prediction/Data/train_V2.csv')
dataset2 = pd.read_csv('E:/MS COMPUTER SCIENCE/MS PROJECTS/OWN PROJECTS/PUBG Chicken Dinner Prediction/Data/test_V2.csv')
df = dataset
testingdf = dataset2 
df.info()

#-----------------READING DATASET------------------------#


#-----------------DATA PREPROCESSING---------------------#

#CHECK FOR ROWS WITH NAN VALUES
df.isnull().sum()

#DROP ROWS WITH NAN IF THE NUMBER OF ROWS IS VERY LOW (OPTIONAL)
df.dropna(subset=['winPlacePerc'], inplace=True)

#DROPPING UNWANTED COLUMNS
df.drop(['Id','groupId','matchId','matchType','winPoints','maxPlace','numGroups','rankPoints','killPlace'], axis=1, inplace=True)
testingdf.drop(['Id','groupId','matchId','matchType','winPoints','maxPlace','numGroups','rankPoints','killPlace'], axis=1, inplace=True)

#REMOVING NAN VALUES FROM THE DATASET
#from sklearn.impute import SimpleImputer as Im
#imputer = Im(missing_values = np.nan, strategy = "mean")
#imputer = imputer.fit(df.iloc[:, 9:11])
#df.iloc[:, 9:11] = imputer.transform(df.iloc[:, 9:11])

#REMOVING SPECIAL CHARACTERS FROM DATA
df['matchType'] = (df['matchType'].str.strip('-'))
testingdf['matchType'] = (testingdf['matchType'].str.strip('-'))

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder
labelencoder_df = LabelEncoder()
df['matchType'] = labelencoder_df.fit_transform(df['matchType']) #Converting matchType To Numerical
testingdf['matchType'] = labelencoder_df.fit_transform(testingdf['matchType']) #Converting matchType To Numerical

#ONEHOT ENCODING OF CATEGORICAL DATA
#df = pd.get_dummies(df, columns=["matchType"], prefix=["matchType"])
#df.drop(['matchType_1','matchType_2','matchType_3','matchType_4','matchType_5','matchType_6','matchType_12','matchType_13'], axis=1, inplace=True)
#testingdf = pd.get_dummies(testingdf, columns=["matchType"], prefix=["matchType"])
#testingdf.drop(['matchType_1','matchType_2','matchType_3','matchType_4','matchType_5','matchType_6','matchType_12','matchType_13'], axis=1, inplace=True)

#PLOT CORRELATION MATRIX AND GRAPH TO FIND DEPENDENT VARIABLES
cor_mat = df.corr()
plot_corr(df)
plot_corr(testingdf)


#-----------------DATA PREPROCESSING---------------------#


#-----------SPLITTING TRAINING AND TESTING DATA----------#

#SEPERATE LABEL COLUMN FROM FEATURE COLUMNS
df_label = df['winPlacePerc'].values
df.drop(['winPlacePerc'], axis=1, inplace=True)
df_feature = df.values

df5k_feature = df_feature[0:5000, :]
df5k_label = df_label[0:5000]

#SPLIT TRAINING SET AND TESTING SET
from sklearn.model_selection._split import train_test_split
feature_train,feature_test,label_train,label_test = train_test_split(df5k_feature, df5k_label, test_size=0.20)


#-----------SPLITTING TRAINING AND TESTING DATA----------#

from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

label_train_encoded = lab_enc.fit_transform(label_train)
print(label_train_encoded)
print(utils.multiclass.type_of_target(label_train))
print(utils.multiclass.type_of_target(label_train.astype('int')))
print(utils.multiclass.type_of_target(label_train_encoded))

label_test_encoded = lab_enc.fit_transform(label_test)
print(label_test_encoded)
print(utils.multiclass.type_of_target(label_test))
print(utils.multiclass.type_of_target(label_train.astype('int')))
print(utils.multiclass.type_of_target(label_test_encoded))


#----------------DATA MODELLING---------------------------#

#NAIVE BAYESIAN
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
clf = GaussianNB()
clf1= MultinomialNB()
clf2=BernoulliNB()
clf.fit(feature_train,label_train_encoded)
clf1.fit(feature_train,label_train_encoded)
clf2.fit(feature_train,label_train_encoded)
accuracy = dict()
predicted_values_GB=clf.predict(feature_test)
predicted_values_NB=clf1.predict(feature_test)
predicted_values_BB=clf2.predict(feature_test)


#GRADIENT BOOSTING ALGORITHM
#from sklearn.ensemble import GradientBoostingClassifier
#model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#model.fit(feature_train, label_train_encoded)
#predicted_values_GBA = model.predict(feature_test)


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(feature_train,label_train_encoded)
model_LR.score(feature_train,label_train_encoded)
#Equation coefficient and Intercept
print('Coefficient: \n', model_LR.coef_)
print('Intercept: \n', model_LR.intercept_)
predicted_values_LR = model_LR.predict(feature_test)

#SVM (SUPPORT VECTOR MACHINE)
from sklearn import svm 
model_svm = svm.SVC()
model_svm.fit(feature_train,label_train_encoded)
model_svm.score(feature_train,label_train_encoded)
predicted_values_svm = model_svm.predict(feature_test)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier()
model_RF.fit(feature_train,label_train_encoded)
predicted_values_RF = model_RF.predict(feature_test)

#LINEAR REGRESSION
#from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
#clf.fit(feature_train,label_train)
#predicted_values_LG = clf.predict(feature_test)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(feature_train,label_train_encoded)
predicted_values_DT = clf.predict(feature_test)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(feature_train,label_train_encoded)
predicted_values_KNN = clf.predict(feature_test)

#----------------DATA MODELLING---------------------------#


#----------------ACCURACY CHECK/CALCULATION---------------------------#

accuracy['Gaussian'] = accuracy_score(predicted_values_GB,label_test_encoded)*100
accuracy['MultinomialNB'] = accuracy_score(predicted_values_NB,label_test_encoded)*100
accuracy['BernoulliNB'] = accuracy_score(predicted_values_BB,label_test_encoded)*100
#accuracy['GBA'] = accuracy_score(predicted_values_GBA,label_test_encoded)*100
accuracy['LogisticReression'] = accuracy_score(predicted_values_LR,label_test_encoded)*100
accuracy['SVM'] = accuracy_score(predicted_values_svm,label_test_encoded)*100
accuracy['RandomForest'] = accuracy_score(predicted_values_RF,label_test_encoded)*100
#accuracy['LinearRegression'] = accuracy_score(predicted_values_LG,label_test_encoded)*100
accuracy['DecisionTree'] = accuracy_score(predicted_values_DT,label_test_encoded)*100
accuracy['KNN'] = accuracy_score(predicted_values_KNN,label_test_encoded)*100
accuracy['Max_accuracy'] = 100
accuracy=pd.DataFrame(list(accuracy.items()),columns=['Algorithm','Accuracy'])
print(accuracy)
sns.lineplot(x='Algorithm',y='Accuracy',data=accuracy)

#----------------ACCURACY CHECK/CALCULATION---------------------------#


#----------------WRITE PREPROCESSED DATA AS A CSV FILE----------------#

#df = pd.DataFrame(X) To convert array to data frame

df.to_csv('train_PUBGDataProcessed.csv', header = None, index = None)
testingdf.to_csv('test_PUBGDataProcessed.csv', header = None, index = None)

#----------------WRITE PREPROCESSED DATA AS A CSV FILE----------------#
