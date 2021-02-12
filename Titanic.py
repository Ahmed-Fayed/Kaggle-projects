# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:03:13 2021

@author: Ahmed Fayed
"""


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from  sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from  sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier





# Acquire data
train_df = pd.read_csv('E:/Software/Kaggle/Datasets/Titanic/train.csv')
test_df = pd.read_csv('E:/Software/Kaggle/Datasets/Titanic/test.csv')
combine = [train_df, test_df]



# Analyze by describing data
print('train ==> ', train_df.shape)
print('test ==> ', test_df.shape)
#print('combine ==> ', combine.shape)

print(train_df.columns.values)
print(train_df.head())
print(train_df.tail())

print(train_df.info())
print('-' * 40)
print(test_df.info()) 

print(train_df.describe())

print(train_df.describe(include=['O']))


print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by = 'Survived', ascending=False))
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))





# Analyze by visualing data
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# correlating numerical nd ordinal features
grid = sns.FacetGrid(train_df, row='Pclass', col='Survived')
grid.map(plt.hist, 'Age', bins=20)
grid.add_legend()

# Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# correlating categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived')
grid.map(sns.barplot, 'Sex', 'Fare', ci=None)
grid.add_legend()




# wrangling data
# Correcting by dropping
print('befor ', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

print('After ', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# creating new feature extracted from existing one
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(combine[0]['Title'], train_df['Sex'])



# Replacing many titles with a more common names or classify them as 'Rare'
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(combine[0][['Title', 'Survived']].groupby(by=['Title'], as_index=False).mean())


















