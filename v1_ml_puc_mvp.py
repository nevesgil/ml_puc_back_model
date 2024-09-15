# -*- coding: utf-8 -*-
"""ml_puc_mvp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15mmW2HoPz2wQI2NBBqc02tZvTGSvr2px

**SPACESHIP TITANIC**

**PUC MVP ENGENHARIA DE SOFTARE**

GILMAR NEVES



---

**INITIAL SETTINGS AND IMPORTS**
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pathlib

filepath = './train.csv'
df = pd.read_csv(filepath)
df.head()

print("Categorical Variables")
categorical_variables = df.select_dtypes(include=['object']).columns
for col in categorical_variables:
    print(col)

print("Numerical Variables")
numerical_variables = df._get_numeric_data().columns
for col in numerical_variables:
    print(col)

df.dtypes

def get_nulls(df):
    dict_nulls = {}
    for col in  df.columns:
        dict_nulls[col]=df[col].isnull().sum()

    df_nulls = pd.DataFrame(data=list(dict_nulls.values()),
                            index=list(dict_nulls.keys()),
                            columns=['#nulls'])
    return df_nulls

get_nulls(df)

def get_nulls_percentage(df):
    dict_nulls = {}
    for col in  df.columns:
        percentage_null_values = str(round(df[col].isnull().sum()/len(df),2))+\
        "%"
        dict_nulls[col] = percentage_null_values

    df_nulls = pd.DataFrame(data=list(dict_nulls.values()),
                            index=list(dict_nulls.keys()),
                            columns=['% nulls'])
    return df_nulls

get_nulls_percentage(df)

for cat_col in categorical_variables:
    if cat_col!="Name":
        df[cat_col] = df[cat_col].fillna(df[cat_col].mode()[0])

for num_col in numerical_variables:
    df[num_col] = df[num_col].fillna(df[num_col].mean())

get_nulls_percentage(df)

df.describe()

df.describe(include=['O'])

# Dropping the name column we don't need anymore
df.drop(['Name'], axis=1, inplace=True)

categorical_variables = df.select_dtypes(include=['object']).columns

categorical_variables

plt.figure(figsize=(10,7))
plt.subplot(2,2,1)
df['HomePlanet'].value_counts().plot(kind='bar', title='HomePlanet')
plt.subplot(2,2,2)
df['Destination'].value_counts().plot(kind='bar', title='Destination')
plt.subplot(2,2,3)
df['CryoSleep'].value_counts().plot(kind='bar', title='CryoSleep')
plt.subplot(2,2,4)
df['VIP'].value_counts().plot(kind='bar',title='VIP')
plt.tight_layout();

df['VIP'].value_counts()

numerical_variables = list(numerical_variables)
if "train" in filepath:
    numerical_variables.remove('Transported')

plt.figure(figsize=(10,7))
for i,num_col in enumerate(numerical_variables):
    plt.subplot(2,3,i+1)
    df[num_col].plot(kind='hist', bins=20)
    plt.title(num_col)
plt.tight_layout();

# building the deck and port features from the 'cabin' column
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df['Port'] = df['Cabin'].apply(lambda s: s[-1] if pd.notnull(s) else 'M')
df["Deck"] = df["Deck"].map({'B':0, 'F':1, 'A':2, 'G':3, 'E':4, 'D':5, 'C':6, 'T':7}).astype(int)
df["Port"] = df["Port"].map({'P':0, 'S':1}).astype(int)
df.drop(['Cabin'], axis=1, inplace=True)
df.head()

df["HomePlanet"] = df["HomePlanet"].map({'Earth':0, 'Europa':1, 'Mars':2}).astype(int)
df.head()

unique_destinations = df["Destination"].unique()
df["Destination"] = df["Destination"].map(dict(zip(unique_destinations,list(range(len(unique_destinations)))))).astype(int)

df.head()

df["Destination"].unique()

if 'train' in filepath:
    df.drop(['PassengerId'],axis=1, inplace=True)

unique_vip = df["VIP"].unique()
df["VIP"] = df["VIP"].map(dict(zip(unique_vip,list(range(len(unique_vip)))))).astype(int)

unique_cryosleep = df["CryoSleep"].unique()
df["CryoSleep"] = df["CryoSleep"].map(dict(zip(unique_cryosleep,list(range(len(unique_cryosleep)))))).astype(int)

filename =  pathlib.Path(filepath).stem + "_cleaned.csv"
file_dest_path = pathlib.Path("./") / filename
df.to_csv(file_dest_path, index=False)

### FIRST TRAINING

df_train = pd.read_csv("./train_cleaned.csv")
# 1. Train Test Split
X = df_train.drop("Transported", axis=1).values
y = df_train["Transported"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape,y_train.shape, X_test.shape, y_test.shape

df_train.head(5)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_score = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_score)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
dtree_score = accuracy_score(y_test, dtree_pred)
print("Decision Tree Accuracy:", dtree_score)

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_score = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_score)

svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_score = accuracy_score(y_test, svc_pred)
print("SVM Accuracy:", svc_score)

results = pd.DataFrame(dict(model=['KNN',
                              'Decision Tree',
                              'Naive Bayes',
                              'SVM'],accuracy=[knn_score, dtree_score,
                                               nb_score, svc_score]))

results

results.plot(kind='bar',x='model',y='accuracy',title='Model Accuracy',legend=False,
        color=['#1F77B4', '#FF7F0E', '#2CA02C'])
plt.ylim(0.5,1);

### Check on the correlation

correlation_matrix=df_train.corr()
correlation_matrix

# Create the correlation heatmap with two decimal places in annotations
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()







