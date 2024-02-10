#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library import
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
#!pip install tensorflow
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#Read data files and provide basic information
data = pd.read_csv("C:/Users/fvuse/OneDrive/Documents/IOT_temp.csv")
# Display 5 random lines
data.sample(5)


# In[3]:


#Unique Values in each column are as follows:
print("Unique values in every column \n"+'-'*25)
for i in data.columns:
    print("\t"+i+" = ",len(set(data[i])))


# In[4]:


#Information on data characteristics
data.info()


# In[5]:


# Description of the data
data.describe()


# In[6]:


# Deletion of data with 'id' and 'room_id/id' labels
df = data.drop(['id','room_id/id'],axis=1)
df.head()


# In[7]:


# Separation of date and time
date=[]
time=[]
for i in df['noted_date']:
    date.append(i.split(' ')[0])
    time.append(i.split(' ')[1])
df['date']=date
df['time']=time


# In[8]:


# Deletion of the 'note_date' label
df.drop('noted_date',axis=1,inplace=True)
df.head()


# In[9]:


# Date separation into days, months and years.
try:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df.drop('date',axis=1,inplace=True)
except:
    print('Operations already performed')
df.head()


# In[10]:


df[['outside','inside']]=pd.get_dummies(df['out/in'])
df.rename(columns = {'out/in':'location'}, inplace = True)


# In[11]:


#Reassembling the database and displaying the new detailed database
df = df[['day','month','year','time','temp','location','outside','inside']]
df.head()


# In[12]:


number_nodes = int(input("Enter the number of nodes to be selected "))


# In[13]:


import random


# In[14]:


indices_nodes = random.sample(range(len(df)), number_nodes)


# In[15]:


selected_nodes = df.iloc[indices_nodes]


# In[16]:


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", selected_nodes)


# In[17]:


#Update the value in the csv file using the replace() function
# updating the column value/data
for i in df.iloc[indices_nodes]:
    
        df.loc[indices_nodes, 'temp'] = df.loc[indices_nodes, 'temp'].replace({29: 15})
    #df.loc[indices_nodes, 'temp'] = df.loc[indices_nodes, 'temp'].replace({41: ''})
    #df.loc[indices_nodes, 'out/in'] = df.loc[indices_nodes, 'out/in'].replace({'In': 'Out'})

# writing into the file
df.to_csv("IOT_temp.csv", index=False)
print(df)


# In[18]:


# Select relevant columns
data_selected = df[["day","month", "time", "temp", "location"]]

max_limit = 51.0  # Define maximum acceptable temperature
min_lit = 25.0  # Define minimum acceptable temperature

# Removing Poison Knots
cleaned_data = df.copy()

# Add an "anomaly" column for temperatures outside the range
cleaned_data['anomaly'] = cleaned_data['temp'].apply(lambda x: 1 if x < min_lit or x > max_limit else 0)

# Separate features
X_cleaned = cleaned_data.drop(columns=["day","month", "time", "location"])

# Training the Isolation Forest model
outlier_model = IsolationForest(contamination=0.07)  
outlier_model.fit(X_cleaned)

# Obtain anomaly detection scores
anomaly_scores = outlier_model.decision_function(X_cleaned)

# Add anomaly detection scores to DataFrame
cleaned_data['anomaly_score'] = anomaly_scores

# Display cleaned DataFrame with anomaly detection scores
print(cleaned_data)

# Calculate average anomaly detection scores
average_anomaly_score = cleaned_data['anomaly_score'].mean()

# Display average anomaly detection scores
print("Average anomaly detection scores :", average_anomaly_score)



# In[20]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Training the Isolation Forest model
outlier_fraction = len(cleaned_data[cleaned_data['anomaly'] == 1]) / len(cleaned_data)
model = IsolationForest(contamination=outlier_fraction)
model.fit(X_cleaned)

# Predicting anomalies

cleaned_data['anomaly_predicted'] = model.predict(X_cleaned)

# Calculate F1 score
# Convert predicted values to 0 (normal) or 1 (abnormal)
cleaned_data['anomaly_predicted'] = [0 if x == 1 else 1 for x in cleaned_data['anomaly_predicted']]
f1 = f1_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])

# Calculate the ROC-AUC curve
roc_auc = roc_auc_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])
fpr, tpr, _ = roc_curve(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])
roc_auc = auc(fpr, tpr)

# Display F1 score and ROC-AUC curve
print("Score F1 :", f1)
print("ROC-AUC :", roc_auc)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rate of false positives')
plt.ylabel('Rate of true positives')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


# In[21]:


from sklearn.metrics import precision_score
# Convert predicted values to 0 (normal) or 1 (abnormal)
cleaned_data['anomaly_predicted'] = [0 if x == 1 else 1 for x in cleaned_data['anomaly_predicted']]

# Calculate precision
precision = precision_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])

# Display precision
print("Precision :", precision)


# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(cleaned_data['anomaly'], cleaned_data['anomaly_score'])

# Drawing the precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')

plt.legend(["Precision-Recall curve"], loc='lower right')

plt.grid(True)
plt.show()


# In[ ]:




