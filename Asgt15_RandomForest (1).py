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

# In[3]:


data = pd.read_csv('C:/Users/17pol/Downloads/Company_Data.csv')
data.head()


# ### EDA

# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# In[7]:


data.describe()


# ### Pairplots

# In[9]:


import seaborn as sns
sns.pairplot(data)


# In[10]:


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


# In[11]:


# checking count of categories for categorical columns colums
sns.countplot(data['ShelveLoc'])
plt.show()

sns.countplot(data['Urban'])
plt.show()

sns.countplot(data['US'])
plt.show()


# ### Applying Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

# In[12]:


data = pd.get_dummies(data)
data.head()


# In[13]:


# Converting Target variable 'Sales' into categories Low, Medium and High.
data['Sales'] = pd.cut(x=data['Sales'],bins=[0, 6, 12, 18], labels=['Low','Medium', 'High'], right = False)
data['Sales']


# In[14]:


data['Sales'].value_counts()


# In[15]:


sns.countplot(data['Sales'])


# In[16]:


data.head()


# In[17]:


dataset = data.values


# In[18]:


# split into input (X) and output (y) variables
X = dataset[:, 1:]
y = dataset[:,0]


# In[19]:


from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

# summarize scores
scores = fit.scores_

features = fit.transform(X)
scores


# In[20]:


data.columns


# In[21]:


col_names = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'Age', 'Education', 'ShelveLoc_Bad', 'ShelveLoc_Good',
       'ShelveLoc_Medium', 'Urban_No', 'Urban_Yes', 'US_No', 'US_Yes']


# In[22]:


score_df = pd.DataFrame(list(zip(scores, col_names)),
               columns =['Score', 'Feature'])
score_df


# In[23]:


data_model = data[['Sales', 'Price', 'Advertising', 'Income', 'Age', 'ShelveLoc_Bad', 'ShelveLoc_Good', 'ShelveLoc_Medium']]
data_model.head()


# In[24]:


X = data_model.iloc[:, 1:]
y = data['Sales']


# ### Splitting data into train and test data

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[26]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Random Forest Classification

# In[28]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

num_trees = 100
max_features = 'auto'
kfold = KFold(n_splits=10)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[29]:


# Train the model on training data
model.fit(x_train, y_train)


# In[30]:


results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# ### Bagged Decision Trees for Classification

# In[32]:



from sklearn.ensemble import BaggingClassifier

kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=42)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# ### AdaBoost Classification

# In[35]:



from sklearn.ensemble import AdaBoostClassifier

kfold = KFold(n_splits=10)
model = AdaBoostClassifier(n_estimators=10)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# In[ ]:




