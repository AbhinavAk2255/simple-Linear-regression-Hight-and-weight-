#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[14]:


a=pd.read_csv('/home/abhinav/Downloads/data.csv')
a


# In[15]:


print(a.isna().sum())


# In[16]:


x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)
x_train


# In[18]:


from sklearn.linear_model import LinearRegression
arg=LinearRegression()
arg.fit(x_train,y_train)
y_pred=arg.predict(x_test)
y_pred


# In[19]:


print(arg.predict([[1.40]]))


# In[20]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.xlabel('height')
plt.ylabel('weight')
plt.show


# In[21]:


at=pd.DataFrame({'Actual value':y_test,'Predicted value':y_pred})
at


# In[22]:


print('slope',arg.coef_)
print('intersept',arg.intercept_)


# In[23]:


from sklearn.metrics import mean_absolute_percentage_error
print('MAE',mean_absolute_percentage_error(y_test,y_pred))


# In[ ]:




