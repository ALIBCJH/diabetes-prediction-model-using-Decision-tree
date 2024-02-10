#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().sum()


# In[6]:


df.columns


# In[7]:


df.duplicated().sum()


# In[8]:


df.isna().sum()


# In[9]:


import seaborn as sns


# In[10]:


import matplotlib.pyplot as plt


# In[15]:


#DATA VISUALIZATION USING SEABORN AKA SNS
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
sns.countplot(df["Outcome"])


# In[22]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Import accuracy_score

df = pd.read_csv("diabetes.csv")

# Define features (X) and target variable (y)
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']  # Assuming 'Outcome' indicates the presence of diabetes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




