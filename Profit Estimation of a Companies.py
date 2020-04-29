#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


companies = pd.read_csv('C:/Users/Rinay Shah/Desktop/Machine Learning/Machine Learning Full/Linear Regression/1000_Companies.csv')
X = companies.iloc[:, :-1].values
Y = companies.iloc[:, 4].values

companies.head()


# In[9]:


#Data Visualization
#Correlation matrix
sns.heatmap(companies.corr())


# In[47]:


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[39]:


X = X[:, 1:]


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[41]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[42]:


y_pred = regressor.predict(X_test)
y_pred


# In[43]:


print(regressor.coef_)


# In[44]:


print(regressor.intercept_)


# In[45]:


from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)


# In[ ]:




