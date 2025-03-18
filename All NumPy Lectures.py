#!/usr/bin/env python
# coding: utf-8

# # lecture 1

# In[1]:


import numpy as np


# In[2]:


array_1d = np.array([1,2,3,4])
print(array_1d)


# In[3]:


type(array_1d)


# In[4]:


array_1d.ndim


# In[5]:


array_1d.size


# In[6]:


array_1d.shape


# In[7]:


array_1d.dtype


# In[8]:


array_2d = np.array([[1,2,3,4],[5,6,7,8]])
print(array_2d)


# In[9]:


type(array_2d)


# In[10]:


array_2d.ndim


# In[11]:


array_2d.size


# In[12]:


array_2d.shape
# 2 = row / 4 columne


# In[13]:


array_2d.dtype


# # lecture 2

# ### Matrix

# In[14]:


#creating matrix by passing values
mx_1s = np.array([[1,1,1],[1,1,1],[1,1,1]])
print(mx_1s)


# In[15]:


#Creating ones matrix(only row) passing single value in argument
mx_1s = np.ones(5)
print(mx_1s)


# In[16]:


mx_1s.dtype


# In[17]:


#Creating ones matrix by passing values in argument (row, columes)
mx_1s = np.ones((3,4))
print(mx_1s)


# In[18]:


#Creating ones matrix and chaging data type passing data type with argument
mx_1s = np.ones((3,4),dtype = int,)
print(mx_1s)


# In[19]:


# Creating zeros matrix
mx_0s = np.zeros((4,6))
print(mx_0s)


# In[20]:


# conveting zero data type in boonian
mx_0s = np.zeros((4,6),dtype = bool)
print(mx_0s)


# In[21]:


# Converting zeros into sting
mx_0s = np.zeros((4,6),dtype = str)
print(mx_0s)


# In[22]:


em_str = ''
print(bool(em_str))


# In[23]:


# creating empty matrix but empty fuction return random values between 1 to 10
empty_mx = np.empty((3,3))
print(empty_mx)


# # lecture 3

# ## Numpy functions :

# In[24]:


import numpy as np


# ### arange()

# In[25]:


# np.arange(start value, End value, steps)
ar_1d = np.arange(1,13)
print(ar_1d)


# In[26]:


even_ar = np.arange(1,13,2)
print(even_ar)


# ### linspace()

# In[27]:


#linspace return random values in given range in stpes - 4
np.linspace(1,5,4)


# ### reshape()

# In[28]:


# Converting 1d array into 2d
ar_2d = ar_1d.reshape(2,6) #(3,4)
print(ar_2d)


# In[29]:


ar_3d = ar_1d.reshape(2,3,2) #(2,3,2) it's a dot product
print(ar_3d)


# In[30]:


# creating 2d ar using arange and reshape
ar = np.arange(1,13).reshape(2,6)
print(ar)


# ### ravel()

# In[31]:


#Ravel convert multidimentional array into 1d
ar.ravel()


# ### flatten()

# In[32]:


#Flatten() convert multi dimentional array into 1d
ar.flatten()


# ### transpose()

# In[33]:


#transpose convert row into colume (this technique use in matrix)
ar.transpose()


# ### T

# In[34]:


#T means transpose()
ar.T


# # lecture 4

# ## Mathematic opration using Numpy

# In[35]:


import numpy as np


# In[36]:


arr1 = np.arange(1,10).reshape(3,3)
arr2 = np.arange(1,10).reshape(3,3)
print(arr1)
print(arr2)


# #### # Addition

# In[37]:


arr1 + arr2


# In[38]:


np.add(arr1,arr2)


# #### # Subtract

# In[39]:


arr1 - arr2


# In[40]:


np.subtract(arr1,arr2)


# #### # Multiplication

# In[41]:


arr1 * arr2


# In[42]:


np.multiply(arr1,arr2)


# #### # Division

# In[43]:


arr1 / arr2


# In[44]:


np.divide(arr1,arr2)


# #### #matrix prodct

# In[45]:


arr1 @ arr2


# In[46]:


arr1.dot(arr2)


# #### # max function

# In[47]:


arr1.max()


# #### # argmax = finding max value index

# In[48]:


arr1.argmax()


# #### # axis = finding max value of each row and column

# In[49]:


arr1.max(axis = 0) # 0 represent row- Horizontal 


# In[50]:


arr1.max(axis = 1) # 1 represent column- vertical


# #### # min = minimun value

# In[51]:


arr1.min()


# In[52]:


arr1.argmin()


# In[53]:


arr1.min(axis = 0)


# #### #sum = adding all elements of matrix

# In[54]:


arr1.sum()


# In[55]:


np.sum(arr1)


# #### #sum and axis = sum of all elements using axis

# In[56]:


np.sum(arr1, axis = 0) # row 


# In[57]:


np.sum(arr1, axis = 1) # column


# #### #mean

# In[58]:


arr1.mean()


# In[59]:


np.mean(arr1)


# In[60]:


np.sqrt(arr1)


# In[61]:


#standerd devision
np.std(arr1)


# In[62]:


#exponent
np.exp(arr1)


# In[63]:


#log
np.log(arr1)


# In[64]:


#log10
np.log10


# # lecture 5

# ## Python numpy array slicing(:)

# In[65]:


import numpy as np


# In[66]:


mx = np.arange(1,101).reshape(10,10)
print(mx)


# In[67]:


mx[0,0] # row, colume


# In[68]:


mx[0,1]


# In[69]:


mx[1,0]


# In[70]:


mx[0,0].ndim # 0 means scaler value


# In[71]:


mx[0]


# In[72]:


mx[:,]


# In[73]:


mx[:,0] #in 1d format


# In[74]:


mx[:,0].ndim


# In[75]:


mx[:,0:1] #in 2d format


# In[76]:


mx[:,0:1].ndim


# In[77]:


mx


# In[78]:


mx[1:4,1:4]


# In[79]:


mx[1:4]


# In[80]:


mx[:,1:3]


# In[81]:


mx[::]


# In[82]:


mx[:,:]


# In[83]:


mx.itemsize #4byte


# In[84]:


mx.dtype # 1byte = 8bit


# In[85]:


32/8


# # lecture 6

# ### Python Numpy Array Conctination split

# In[86]:


import numpy as np


# In[87]:


arr1 = np.arange(1,17).reshape(4,4)
print(arr1)


# In[88]:


arr2 = np.arange(17,33).reshape(4,4)
print(arr2)


# #### concatination

# In[89]:


list1 = [2,4,6,7]
list2 = [1,8,3,5]


# In[90]:


list1 + list2


# In[91]:


arr1 + arr2 #failed to concatinate


# In[92]:


np.concatenate((arr1,arr2))


# In[93]:


print(arr1)
print(arr2)


# In[94]:


np.concatenate((arr1,arr2),axis = 1)
# in axis by default value is 0 mean colume and 1 mean row.


# In[95]:


np.concatenate((arr1,arr2),axis = 0)


# In[96]:


#stack
np.vstack((arr1,arr2))


# In[97]:


np.hstack((arr1,arr2)) #horizontal = row wise concate


# In[98]:


np.hstack((arr1,arr2,arr1)) #can concate multipe matrix


# #### split

# In[99]:


arr1


# In[100]:


np.split(arr1,2)


# In[101]:


list1 = np.split(arr1,2)
print(list1)


# In[102]:


type(list1)


# In[103]:


list1[0]


# In[104]:


type(list1[0]) # 0 th value datatype


# In[105]:


#split row vise
np.split(arr1,2,axis = 1)


# In[106]:


#split with 1d array
d1 = np.array([4,5,1,3,9])


# In[107]:


np.split(d1,[1,3]) #index


# In[108]:


#1 return 4 = return before 1th index value means 4 
#3 return [5,1] = return before 3rd index values mean 5, 1
#black space return remaining values


# In[ ]:





# # lecture 7

# ### Python NumPy Tutorial :8
# 
# ### Find Trignomentry sin(), cos() and tan() using NumPy Trignometry functions

# In[109]:


import numpy as np


# In[110]:


import matplotlib.pyplot as plt


# In[111]:


np.sin(180) # 180 is angle value #out put will in degree
# if you wnt output in radian then multiply with pie value


# In[112]:


# A radian is a unit of measurement for ~angles~ in the international system of unit(si)
np.sin(180*np.pi/180)


# In[113]:


np.sin(190)


# In[114]:


np.cos(180)


# In[115]:


np.tan(180)


# #### visualizing sin cos tan
# 

# In[116]:


# for visualizing sin cos tan we need two values y and x, 
# so we have to create array


# In[117]:


x_sin = np.arange(0,3*np.pi,0.1) #np.pi means pie value 3.14 #0.1 is step value
print(x_sin)


# In[118]:


y_sin = np.sin(x_sin)
print(y_sin)


# In[119]:


plt.plot(x_sin,y_sin)
plt.show()


# In[120]:


# generating cos value
y_cos = np.cos(x_sin)
plt.plot(x_sin,y_cos)
plt.show


# In[121]:


# generating tan value
y_tan = np.tan(x_sin)
plt.plot(x_sin,y_tan)
plt.show


# In[ ]:





# # lecutre 8

# ## Random Sampling with NumPy

# In[122]:


import numpy as np


# In[123]:


import random


# In[124]:


np.random.random(1) 
#random.random will return value between only 0 to 1.


# In[125]:


# Creating 2d array using random function
np.random.random((3,3)) #R C


# In[126]:


# Creating integer value scaler/vector using random fuction
np.random.randint(1,4)


# In[127]:


# Creating integer value 2d/3d array using random fuction
np.random.randint(1,4, (4,4))
#(4,4) is shape


# In[128]:


# Creating 3d array
np.random.randint(1,4, (2,4,4))


# In[129]:


# Creating 4d array
np.random.randint(1,4, (2,2,4,4))


# In[130]:


# We get random values while creating array using random but we need same value use seed
#seed

np.random.seed(10)
np.random.randint(1,4,(2,4,4))


# In[131]:


#creating another we get same values (seed value is 10)
np.random.seed(10)
np.random.randint(1,4,(2,4,4))


# In[132]:


2**32-1


# In[133]:


#rand = we can create 1d array

np.random.rand()


# In[134]:


np.random.rand(3)


# In[135]:


np.random.rand(3,3)


# In[136]:


#randn = return negative and positve value
np.random.randn(3,3)


# In[137]:


#if we want any item from sequence use choice function
x = [1,2,3,4]
np.random.choice(x)


# In[138]:


x = [1,2,3,4]
np.random.choice(x)


# In[139]:


x = [1,2,3,4]
np.random.choice(x)


# In[140]:


for i in range(20):
    print(np.random.choice(x))


# #### Permutation

# In[141]:


# Permutation means the arrangement of a set of objects in a specific order, or the process of chaging the order of an existing set


# In[142]:


x


# In[143]:


np.random.permutation(x)


# In[144]:


np.random.permutation(x) # we can use shuffle insted of permutation


# In[ ]:





# # lecture 10

# ## String Operations, Comparison and information

# In[145]:


import numpy as np


# In[146]:


# Generating two string
ch_name = "Swaraj AI Production"
str1 = "Learning Python Numpy"


# In[147]:


#performing oprations on string add
np.char.add(ch_name,str1)


# In[148]:


#lower
np.char.lower(ch_name)


# In[149]:


np.char.upper(ch_name)


# In[150]:


np.char.center(str1,60)


# In[151]:


np.char.center(str1,90,fillchar ="-")


# In[152]:


np.char.split(ch_name)


# In[153]:


np.char.splitlines("hello\nIndian") #\n


# In[154]:


str4 = "dmy"
str5 = "dmy"


# In[155]:


np.char.join([":","/"],[str4,str5])


# In[156]:


np.char.replace(ch_name,"AI","Artificial Intelligence")


# In[157]:


np.char.equal(str4,str5) #true means same #noequal #less


# In[158]:


np.char.count(ch_name,"a")


# In[159]:


ch_name


# In[160]:


np.char.find(ch_name,"AI")


# In[ ]:





# In[ ]:




