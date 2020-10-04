#!/usr/bin/env python
# coding: utf-8

# Name= Umesh Karamchandani |
# Assignment 4

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("dataset.csv")


# In[120]:


df.head(5)


# In[8]:


df1=df[["x01","x01","x03","x04","x05","x06","x07","x08","x09","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","x21","x22","x23","x24","x29","x30","x31","x32","x33","x34","x35","x36","x37","x38"]]


# In[121]:


df1


# In[17]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[34]:


scaler_x=preprocessing.StandardScaler().fit_transform(df1)


# In[35]:


scaler_x[:,0]


# In[182]:


from sklearn.decomposition import PCA


# In[191]:


pca=PCA().fit(scaler_x)
pca.explained_variance_
   
    


# In[194]:


np.set_printoptions(suppress=True)
pca.explained_variance_ratio_


# In[216]:


print(pca.explained_variance_ratio_.cumsum())
for j,i in enumerate(pca.explained_variance_ratio_.cumsum()):
    if i >0.85:
        print(j+1)
        break
for j,i in enumerate(pca.explained_variance_ratio_.cumsum()):
    if i >0.99:
        print(j+1)
        break


# 9 Features would be required to obtain 86.2 % Variance 
#  16 Features would be required to obtain 99 % Variance

# In[207]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))


# In[186]:


pca=PCA(n_components=2)


# In[187]:


pca.fit(scaler_x)


# In[188]:


x_pca=pca.transform(scaler_x)


# In[209]:


x_pca.shape


# In[206]:


plt.figure(figsize=(14,8))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df["Target"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# Answer 2) As per the above Scatter Plot Highest Variance is along the Prinicpal Component_1 follwed by Principal Component 2.
# 

# In[ ]:




