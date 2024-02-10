#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 Importing a library
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


#nrows_read = 100000 # specify 'None' if want to read whole file (405,184 rows)
df = pd.read_csv("D:\Données_Thèse\Articles/iot_telemetry_data.csv")
df.head()


# In[3]:


# The last 5 lines
df.tail()


# In[4]:


#The form of our data is as follows:
print("The form of our data is : ",df.shape)


# In[5]:


#The unique Values in each column are as follows:
print("The unique Values in each column \n"+'-'*25)
for i in df.columns:
    print("\t"+i+" = ",len(set(df[i])))


# In[6]:


#Information on data characteristics
df.info()


# In[7]:


# Deletion of data with labels 'id' and 'room_id/id'
df = df.drop(['co','light','lpg','motion','temp','smoke'],axis=1)
df.head()


# In[8]:


#a)Null data
df.isnull().sum()


# In[9]:


# Humidity data values
print("Humidity -> \n"+"-"*40)
print("\tTotal Count    = ",df['humidity'].shape[0])
print("\tMinimum Value  = ",df['humidity'].min())
print("\tMaximum Value  = ",df['humidity'].max())
print("\tMean Value     = ",df['humidity'].mean())
print("\tStd dev Value  = ",df['humidity'].std())
print("\tVariance Value = ",df['humidity'].var())


# In[10]:


nombre_noeuds = int(input("Enter the number of nodes to select : "))


# In[11]:


# Display number of selected nodes
print("The selected nodes are : ",nombre_noeuds )


# In[12]:


import random


# In[13]:


indices_noeuds = random.sample(range(len(df)), nombre_noeuds)


# In[14]:


noeuds_selectionnes = df.iloc[indices_noeuds]


# In[15]:


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", noeuds_selectionnes)


# In[16]:


# Updating the column value/data
import numpy as np
for i in df.iloc[indices_noeuds]:
    rows_to_modify = int(0.3 * len(df))
indices_to_modify = np.random.choice(df.index, rows_to_modify, replace=False)
df.loc[indices_to_modify, 'humidity'] = df.loc[indices_to_modify, 'humidity'] *0.5  
# writing into the file
df.to_csv("iot_telemetry_data.csv", index=False)
print(df)


# In[18]:


# Select columns relevant to anomaly detection ("ts", "device", "humidity")
selected_columns = ["ts","device", "humidity"]
data_selected = df[selected_columns]
# Remove missing values if necessary
data_selected = data_selected.dropna()
# Data pre-processing
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  # Normalize values between 0 and 1
data_normalized = scaler.fit_transform(data_selected["humidity"].values.reshape(-1, 1))
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
# Model drive
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Detecting compromised nodes
predictions = model.predict(X_train)
anomalies = np.where(abs(predictions - y_train) > 0.47) 
# Displaying compromised nodes
compromised_nodes = data_selected.iloc[anomalies[0]]
print(compromised_nodes)


# In[19]:


# Displaying compromised nodes in a graph
plt.figure(figsize=(8, 5))
plt.plot(data_selected.index, data_selected["humidity"], label="Temperature data")
plt.scatter(data_selected.index[anomalies[0]], data_selected["humidity"].iloc[anomalies[0]], color='red', label="Compromised nodes")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Detection of compromised nodes (poisoned model)")
plt.legend()
plt.show()


# In[19]:


# Select relevant columns for anomaly detection ("ts","device", "humidity")
selected_columns = ["ts","device", "humidity"]
data_selected = df[selected_columns]
# Remove missing values if necessary
data_selected = data_selected.dropna()
# Data pre-processing
scaler = MinMaxScaler()  #  # Normalize values between 0 and 1
data_normalized = scaler.fit_transform(data_selected["humidity"].values.reshape(-1, 1))
# Creating the Isolation Forest model
model = IsolationForest(contamination=0.05)  # 5% de contamination

# Model drive
model.fit(data_normalized)

# Anomaly prediction
anomalies = model.predict(data_normalized)
data_selected["anomaly"] = anomalies
# Display potentially poisonous nodes
poisoned_nodes = data_selected[data_selected["anomaly"] == -1]
print(poisoned_nodes)


# In[20]:


# Displaying compromised nodes in a graph
plt.figure(figsize=(10, 6))
plt.scatter(data_selected.index, data_selected["humidity"], c=data_selected["anomaly"], cmap="viridis", label="Normal nodes")
plt.scatter(data_selected.index[data_selected["anomaly"] == -1], data_selected["humidity"][data_selected["anomaly"] == -1], color='red', label="Compromise nodes")
plt.xlabel("Data index")
plt.ylabel("Humidity")
plt.title("Detection of compromised nodes with Isolation Forest (poisoned data)")
plt.legend()
plt.show()


# In[21]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[22]:


# Check and clean the "temperature" column
df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
df.dropna(subset=['humidity'], inplace=True)


# In[23]:


# Remove poisoned nodes (anomalous data) using residue-based anomaly detection
# Use 'temperature' columns to detect anomalies
X = df[['humidity']]


# In[24]:


# Create an anomaly detection model based on Isolation Forest
outlier_model = IsolationForest(contamination=0.05)
outlier_model.fit(X)
# Predicting anomalies
anomalies = outlier_model.predict(X)
df['anomaly'] = np.where(anomalies == -1, 1, 0)

# Remove poisoned nodes (abnormal data)
data_filtered = df[df['anomaly'] == 0]


# In[25]:


from sklearn.model_selection import train_test_split
# Divide data into training and test sets
X = data_filtered[['humidity']]
y = data_filtered['anomaly']

# Separate data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use 'temperature' columns to detect anomalies
X_train = X_train[['humidity']]
X_test = X_test[['humidity']]

# Create a new Isolation Forest model
outlier_model_new = IsolationForest(contamination=0.05)
outlier_model_new.fit(X_train)

# Predict anomalies for the test set
anomalies_new = outlier_model_new.predict(X_test)


# In[17]:


# Select relevant columns
data_selected = df[["ts","device", "humidity"]]


min_lit = 35.0  # Define minimum acceptable temperature

# Removing Poison Knots
cleaned_data = df.copy()


# In[19]:


# Add an "anomaly" column for temperatures outside the range
cleaned_data['anomaly'] = cleaned_data['humidity'].apply(lambda x: 1 if x < min_lit else 0)


# In[20]:


# Separate features
X_cleaned = cleaned_data.drop(columns=["ts","device"])


# In[21]:


# Training the Isolation Forest model
outlier_model = IsolationForest(contamination=0.12)  
outlier_model.fit(X_cleaned)


# In[22]:


# Obtain anomaly detection scores
anomaly_scores = outlier_model.decision_function(X_cleaned)
# Add anomaly detection scores to DataFrame
cleaned_data['anomaly_score'] = anomaly_scores


# In[23]:


# Display cleaned DataFrame with anomaly detection scores
print(cleaned_data)


# In[24]:


# Calculate average anomaly detection scores
average_anomaly_score = cleaned_data['anomaly_score'].mean()
# Display average anomaly detection scores
print("Average anomaly detection scores :", average_anomaly_score)


# In[26]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Training the Isolation Forest model
outlier_fraction = len(cleaned_data[cleaned_data['anomaly'] == 1]) / len(cleaned_data)
model = IsolationForest(contamination=outlier_fraction)
model.fit(X_cleaned)


# In[27]:


# Predicting anomalies

cleaned_data['anomaly_predicted'] = model.predict(X_cleaned)


# In[28]:


# Calculate F1 score
# Convert predicted values to 0 (normal) or 1 (abnormal)
cleaned_data['anomaly_predicted'] = [0 if x == 1 else 1 for x in cleaned_data['anomaly_predicted']]
f1 = f1_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])


# In[29]:


# Calculate the ROC-AUC curve
roc_auc = roc_auc_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])
fpr, tpr, _ = roc_curve(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])
roc_auc = auc(fpr, tpr)


# In[30]:


# Display F1 score and ROC-AUC curve
print("Score F1 :", f1)
print("ROC-AUC :", roc_auc)


# In[25]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

# Calculer la fraction d'anomalies dans votre ensemble de données
outlier_fraction = len(cleaned_data[cleaned_data['anomaly'] == 1]) / len(cleaned_data)

# Créer le modèle IsolationForest avec la valeur 'auto' pour le paramètre contamination
model = IsolationForest(contamination='auto')

# Adapter le modèle
model.fit(X_cleaned)


# In[31]:


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


# In[32]:


from sklearn.metrics import precision_score
# Convert predicted values to 0 (normal) or 1 (abnormal)
cleaned_data['anomaly_predicted'] = [0 if x == 1 else 1 for x in cleaned_data['anomaly_predicted']]

# Calculate precision
precision = precision_score(cleaned_data['anomaly'], cleaned_data['anomaly_predicted'])

# Display precision
print("Precision :", precision)

