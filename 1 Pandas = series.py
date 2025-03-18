#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


#Series = 1d lable homogens array
#dataFrame = 2d dimentional lable hetrogeniously tabuler structure
#panel = 3d dimentional lable array


# ### pandas series

# In[3]:


import pandas as pd


# In[4]:


pd.__version__


# #### creating series using list

# In[5]:


list_s = [1,1,2,-3,6.2,'data values']
print(list_s)


# In[6]:


series1 = pd.Series(list_s)
print(series1)


# In[7]:


type(series1)


# #### creating short series shortcut

# In[8]:


series2 = pd.Series([1,2,3,4])
print(series2)


# ### Empty list

# In[11]:


empty_s = pd.Series([])
print(empty_s)


# ### chaging index

# In[12]:


series3 = pd.Series([1,2,3,4],index = ['A','B','C','D'])
print(series3)


# ### changing data type

# In[13]:


series3 = pd.Series([1,2,3,4],index = ['A','B','C','D'], dtype = float)
print(series3)


# ### changing series name

# In[14]:


series3 = pd.Series([1,2,3,4],index = ['A','B','C','D'], dtype = float, name = 'data value')
print(series3)


# ### creating series using scaler value

# In[15]:


scalar_s = pd.Series(0.5)
print(scalar_s)


# In[16]:


scalar_s = pd.Series(0.5,index = [1,2,3])
print(scalar_s)


# ### creating series using dictionary
# 

# In[17]:


dict_s = pd.Series({'a':1,'s':2})
print(dict_s)


# ### acessing values from series

# In[18]:


s4 = pd.Series([1,2,3,4,5])
print(s4)


# In[19]:


s4[0]


# In[20]:


s4[1]


# In[21]:


s4[0:3]


# In[22]:


max(s4)


# In[23]:


min(s4)


# In[24]:


s4[s4>3]


# In[25]:


s4


# ### mathematical opration

# In[26]:


s4


# In[27]:


s5 = pd.Series([1,2,3,4,5])
print(s5)


# In[28]:


s5+s4


# In[29]:


s6 = pd.Series([1,2,3])
print(s6)


# In[30]:


s6+s5


# In[ ]:




