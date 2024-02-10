#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 Library import
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


#Display first 5 lines
data.head()


# In[4]:


# The last 5 lines
data.tail()


# In[5]:


#The form of our data is as follows
print("The form of our data is: ",data.shape)


# In[6]:


#Unique Values in each column are as follows:
print("Unique values in every column \n"+'-'*25)
for i in data.columns:
    print("\t"+i+" = ",len(set(data[i])))


# In[7]:


# Count the number of "X" values in the column
nombre_X = data['out/in'].value_counts()['Out']

print(nombre_X)


# In[8]:


#Information on data characteristics
data.info()


# In[9]:


# Description of the data
data.describe()


# In[10]:


# Deletion of data with 'id' and 'room_id/id' labels
df = data.drop(['id','room_id/id'],axis=1)
df.head()


# In[11]:


# Null data
data.isnull().sum()


# In[12]:


#Separation of date and time
date=[]
time=[]
for i in df['noted_date']:
    date.append(i.split(' ')[0])
    time.append(i.split(' ')[1])
df['date']=date
df['time']=time


# In[13]:


# Deletion of the 'note_date' label
df.drop('noted_date',axis=1,inplace=True)
df.head()


# In[14]:


df[['outside','inside']]=pd.get_dummies(df['out/in'])
df.rename(columns = {'out/in':'location'}, inplace = True)


# In[15]:


print('Total Inside Observations  :',len([i for i in df['inside'] if  i == 1]))
print('Total Outside Observations :',len([i for i in df['inside'] if  i == 0]))


# In[16]:


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


# In[17]:


print("Days of observation   : ",sorted(df['day'].unique()))
print("Months of observation : ",sorted(df['month'].unique()))
print("Year of observation   : ",sorted(df['year'].unique()))


# In[18]:


# Temperature data values
print("Temperature -> \n"+"-"*30)
print("\tTotal Count    = ",df['temp'].shape[0])
print("\tMinimum Value  = ",df['temp'].min())
print("\tMaximum Value  = ",df['temp'].max())
print("\tMean Value     = ",df['temp'].mean())
print("\tStd dev Value  = ",df['temp'].std())
print("\tVariance Value = ",df['temp'].var())


# In[19]:


#Reassembling the database and displaying the new detailed database
df = df[['day','month','year','time','temp','location','outside','inside']]
df.head()


# In[20]:


number_nodes = int(input("Enter the number of nodes to be selected : "))


# In[21]:


# Display number of selected nodes
print("The number of randomly selected nodes is : ",number_nodes )


# In[22]:


import random


# In[23]:


indices_nodes = random.sample(range(len(df)), number_nodes)


# In[24]:


selected_nodes = df.iloc[indices_nodes]


# In[25]:


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", selected_nodes)


# In[26]:


#Update the value in the csv file using the replace() function
# updating the column value/data
for i in df.iloc[indices_nodes]:
    
    #df.loc[indices_nodes, 'outside'] = df.loc[indices_nodes, 'inside'].replace()
    #df.loc[indices_nodes, 'inside'] = df.loc[indices_nodes, 'outside'].replace()
    #df.loc[indices_nodes, 'room_id/id'] = df.loc[indices_nodes, 'out/in'].replace()
    df.loc[indices_nodes, 'temp'] = df.loc[indices_nodes, 'temp'].replace({29: 15})
    #df.loc[indices_nodes, 'temp'] = df.loc[indices_nodes, 'temp'].replace({41: ''})
    #df.loc[indices_nodes, 'out/in'] = df.loc[indices_nodes, 'out/in'].replace({'In': 'Out'})

# writing into the file
df.to_csv("IOT_temp.csv", index=False)
print(df)


# In[27]:


# Temperature data values
print("Temperature -> \n"+"-"*30)
print("\tTotal Count    = ",df['temp'].shape[0])
print("\tMinimum Value  = ",df['temp'].min())
print("\tMaximum Value  = ",df['temp'].max())
print("\tMean Value     = ",df['temp'].mean())
print("\tStd dev Value  = ",df['temp'].std())
print("\tVariance Value = ",df['temp'].var())


# In[27]:


print(data.columns)


# In[28]:


print(df.columns)


# In[29]:


# Select relevant columns for anomaly detection (Id,time, temp,location)
selected_columns = ["day","month", "time", "temp", "location"]
data_selected = df[selected_columns]
# Remove missing values if necessary
data_selected = data_selected.dropna()
# Data pre-processing
scaler = MinMaxScaler()  # Normalize values between 0 and 1
data_normalized = scaler.fit_transform(data_selected["temp"].values.reshape(-1, 1))
# Preparing data for learning
X_train = data_normalized[:-1]
y_train = data_normalized[1:]
# Creating the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
# Model compilation
model.compile(optimizer='adam', loss='mse')
# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Detecting compromised nodes
predictions = model.predict(X_train)
anomalies = np.where(abs(predictions - y_train) > 0.47)  
# Displaying compromised nodes
compromised_nodes = data_selected.iloc[anomalies[0]]
print(compromised_nodes)


# In[30]:


# Displaying compromised nodes in a graph
plt.figure(figsize=(8, 5))
plt.plot(data_selected.index, data_selected["temp"], label="Temperature data")
plt.scatter(data_selected.index[anomalies[0]], data_selected["temp"].iloc[anomalies[0]], color='red', label="Compromise nodes")
plt.xlabel("Data index")
plt.ylabel("Temperature")
plt.title("Detection of compromised nodes (poisoned data)")
plt.legend()
plt.show()


# In[31]:


# Select relevant columns for anomaly detection (Id,time, temp,location)
selected_columns = ["day","month","year", "time", "temp", "location"]
data_selected = df[selected_columns]
# Remove missing values if necessary
data_selected = data_selected.dropna()
# Data pre-processing
scaler = MinMaxScaler()  #  # Normalize values between 0 and 1
data_normalized = scaler.fit_transform(data_selected["temp"].values.reshape(-1, 1))
# Creating the Isolation Forest model
model = IsolationForest(contamination=0.12)  # 2% de contamination

# Model drive
model.fit(data_normalized)

# Anomaly prediction
anomalies = model.predict(data_normalized)
data_selected["anomaly"] = anomalies
# Display potentially poisonous nodes
poisoned_nodes = data_selected[data_selected["anomaly"] == -1]
print(poisoned_nodes)


# In[32]:


# Displaying compromised nodes in a graph
plt.figure(figsize=(10, 6))
plt.scatter(data_selected.index, data_selected["temp"], c=data_selected["anomaly"], cmap="viridis", label="Normal nodes")
plt.scatter(data_selected.index[data_selected["anomaly"] == -1], data_selected["temp"][data_selected["anomaly"] == -1], color='red', label="Compromise nodes")
plt.xlabel("Data index")
plt.ylabel("Temperature")
plt.title("Detection of compromised nodes with Isolation Forest (poisoned data)")
plt.legend()
plt.show()


# In[33]:


# Check and clean the "temperature" column
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df.dropna(subset=['temp'], inplace=True)


# In[34]:


# Remove poisoned nodes (anomalous data) using residue-based anomaly detection
# Use 'temperature' columns to detect anomalies
X = df[['temp']]


# In[35]:


# Create an anomaly detection model based on Isolation Forest
outlier_model = IsolationForest(contamination=0.12)
outlier_model.fit(X)
# Predicting anomalies
anomalies = outlier_model.predict(X)
df['anomaly'] = np.where(anomalies == -1, 1, 0)

# Remove poisoned nodes (abnormal data)
data_filtered = df[df['anomaly'] == 0]


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


# Divide data into training and test sets
X = data_filtered[['temp']]
y = data_filtered['anomaly']

# Separate data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use 'temperature' columns to detect anomalies
X_train = X_train[['temp']]
X_test = X_test[['temp']]

# Create a new Isolation Forest model
outlier_model_new = IsolationForest(contamination=0.12)
outlier_model_new.fit(X_train)

# Predict anomalies for the test set
anomalies_new = outlier_model_new.predict(X_test)


# In[38]:


from sklearn.metrics import log_loss


# In[39]:


# Filter test data to keep only non-abnormal data
X_test_filtered = X_test[anomalies_new == 1]
y_test_filtered = y_test[anomalies_new == 1]

# Reset initial test set index
X_test.reset_index(drop=True, inplace=True)

# Get the anomalous density scores for each sample in the test set
anomaly_scores = outlier_model_new.score_samples(X_test_filtered)

# Create a list of labels with two classes (0 for normal class and 1 for anomalies)
labels = [0, 1]

# Calculate the confidence score (log loss) for the anomaly detection model
confidence_score = log_loss(y_test_filtered, anomaly_scores, labels=labels)
print("Confidence score (log loss) for the anomaly detection model :", confidence_score)



# In[41]:


# Density plot to represent the density of anomaly detection scores in the test set
plt.figure(figsize=(8, 5))
plt.hist(anomaly_scores, bins=50, density=True, alpha=0.6, color='blue')
plt.xlabel('Anomaly Detection Score')
plt.ylabel('Density')
plt.title('Density plot of anomaly detection scores in the test set')
plt.show()


# In[42]:


# Obtaining the index of abnormal samples in the initial test set
anomaly_indices = X_test[anomalies_new == -1].index

# Removing abnormal samples from the new DataFrame
data_cleaned = df.drop(anomaly_indices)

# Displaying the new DataFrame without the abnormal samples
print(data_cleaned)

# Saving the new cleaned DataFrame to a new CSV file
data_cleaned.to_csv('C:/Users/fvuse/OneDrive/Documents/nettoye_1.csv', index=False)


# In[43]:


# Separate features and labels for the new cleaned dataset
X_cleaned = data_cleaned.drop(columns=['day', 'month', 'year','time','location','outside','inside', 'anomaly'])
y_cleaned = data_cleaned['anomaly']

# Fraction of anomalies (you can adjust this value according to your case)
outlier_fraction = 0.01

# Train a new anomaly detection model on the cleaned dataset
outlier_model_cleaned = IsolationForest(contamination=outlier_fraction)
outlier_model_cleaned.fit(X_cleaned)

# Get the confidence scores for the new cleaned test set
anomaly_scores_cleaned = outlier_model_cleaned.score_samples(X_cleaned)

# Calculate the confidence score (log loss) for the anomaly detection model
confidence_score_cleaned = log_loss(y_cleaned, -anomaly_scores_cleaned)  
print("Confidence score (log loss) for anomaly detection model after anomaly removal :", confidence_score_cleaned)


# In[ ]:




