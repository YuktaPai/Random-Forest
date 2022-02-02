#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# ### Data

# In[4]:


# Loading Dataset
data = pd.read_csv('C:/Users/17pol/Downloads/Fraud_check.csv')
data.head()


# ### EDA & Data Preprocessing

# In[5]:


data.shape


# In[6]:


data.info()


# In[8]:


data.isna().sum()


# In[10]:


data.sample(10)


# In[11]:


# Renaming columns
data = data.rename({'Undergrad':'under_grad', 'Marital.Status':'marital_status', 'Taxable.Income':'taxable_income',
                    'City.Population':'city_population', 'Work.Experience':'work_experience', 'Urban':'urban'}, axis = 1)
data.head()


# In[12]:


data.describe()


# In[13]:


# checking count of categories for categorical columns colums
import seaborn as sns

sns.countplot(data['under_grad'])
plt.show()

sns.countplot(data['marital_status'])
plt.show()

sns.countplot(data['urban'])
plt.show()


# In[14]:


# Checking for outliers in numerical data
sns.boxplot(data['taxable_income'])
plt.show()

sns.boxplot(data['city_population'])
plt.show()

sns.boxplot(data['work_experience'])
plt.show()


# In[15]:


# Correlation analysis for data
corr = data.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 6))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[16]:


# Converting categorical variables into dummy variables
data = pd.get_dummies(data)
data.head()


# In[17]:


# Converting taxable_income <= 30000 as "Risky" and others are "Good"
data['taxable_category'] = pd.cut(x = data['taxable_income'], bins = [10002,30000,99620], labels = ['Risky', 'Good'])
data.head()


# In[18]:


sns.countplot(data['taxable_category'])


# In[19]:


data['taxable_category'].value_counts()


# In[20]:


# dropping column taxable_income
data1 = data.drop('taxable_income', axis = 1)
data1


# In[21]:


# Correlation analysis for data11
corr = data1.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 6))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[22]:


# Dividing data into independent variables and dependent variable
X = data1.drop('taxable_category', axis = 1)
y = data1['taxable_category']
X


# In[23]:


y


# ### Splitting data into train and test data

# In[24]:



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Random Forest Classification

# In[25]:



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[26]:


num_trees = 100
max_features = 'auto'
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[28]:


# Train the model on training data
model.fit(x_train, y_train)
RandomForestClassifier()
kfold = KFold(n_splits=10)

results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# ### Bagged Decision Trees for Classification

# In[30]:



from sklearn.ensemble import BaggingClassifier

kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=42)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# ### AdaBoost Classification

# In[32]:



from sklearn.ensemble import AdaBoostClassifier

kfold = KFold(n_splits=10)
model = AdaBoostClassifier(n_estimators=10)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# In[ ]:





# In[ ]:




