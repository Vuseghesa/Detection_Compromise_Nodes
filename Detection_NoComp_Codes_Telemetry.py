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


number_nodes = int(input("Enter the number of nodes to select : "))


# In[11]:


import random


# In[13]:


indices_nodes = random.sample(range(len(df)), number_nodes)


# In[14]:


nodes_selected = df.iloc[indices_nodes]


# In[15]:


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", nodes_selected)


# In[16]:


# Updating the column value/data
import numpy as np
for i in df.iloc[indices_nodes]:
    rows_to_modify = int(0.3 * len(df))
indices_to_modify = np.random.choice(df.index, rows_to_modify, replace=False)
df.loc[indices_to_modify, 'humidity'] = df.loc[indices_to_modify, 'humidity'] *0.5  
# writing into the file
df.to_csv("iot_telemetry_data.csv", index=False)
print(df)


# In[17]:


from sklearn.model_selection import train_test_split

# Select relevant columns
data_selected = df[["ts","device", "humidity"]]

seuil_max = 75.0  # set maximum acceptable humidity
seuil_min = 45.0  # set the maximum acceptable humidity

# Add an "anomaly" column for humidity outside the range
data_selected['anomaly'] = data_selected['humidity'].apply(lambda x: 0 if x < seuil_min or x > seuil_max else 1)

# Remove missing values 
data_selected = data_selected.dropna()
# Data pre-processing
scaler = MinMaxScaler()  # Normalize values between 0 and 1
data_normalized = scaler.fit_transform(data_selected["anomaly"].values.reshape(-1, 1))


# Preparing data for learning
X = data_normalized[:-1]
y = data_normalized[1:]

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, accuracy: {accuracy}")

# Test set prediction
predictions = model.predict(X_test)

# Converting predictions into binary classes 
predictions_binary = (predictions > 0.5).astype(int)


# In[18]:


# Tracing accuracy through the ages
plt.figure(figsize=(7, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and validation Accuracy with compromised nodes')
plt.xlabel('Epoch')
plt.ylabel(f"accuracy: {accuracy*100}")
plt.legend()
plt.show()


# In[19]:


from sklearn.metrics import accuracy_score
# Calculating accuracy
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy}")


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_binary)

# Classification report
class_report = classification_report(y_test, predictions_binary)

# Confusion matrix display
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion matrix")
plt.xlabel("Predictions")
plt.ylabel("True values")
plt.show()

# Displaying the classification report
print("Classification report :")
print(class_report)


# In[21]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[22]:


# Test set prediction
predictions = model.predict(X_test)
# Calculate the F1 score
f1 = f1_score(y_test, predictions_binary)
print(f1)


# In[23]:


from sklearn.metrics import precision_score
precision = precision_score(y_test, predictions_binary)
print(precision)


# In[24]:


fpr, tpr, thresholds = roc_curve(y_test, predictions_binary)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[25]:


# Obtaining anomaly indexes
anomaly_indices = [i for i in range(len(predictions_binary)) if predictions_binary[i] == 0]

# Delete the lines corresponding to anomalies in your test set
X_test_no_anomalies = np.delete(X_test, anomaly_indices, axis=0)
y_test_no_anomalies = np.delete(y_test, anomaly_indices, axis=0)


# In[26]:


# Creating of a new pandas DataFrame
df_no_anomalies = pd.DataFrame(X_test_no_anomalies, columns=['anomaly'])
df_no_anomalies['cleandata'] = y_test_no_anomalies


# In[27]:


# Prediction on the new test set without anomalies
new_predictions = model.predict(X_test_no_anomalies)

# Converting new predictions into classes 
new_predictions_binary = (new_predictions > 0.5).astype(int)

precision_new = precision_score(y_test_no_anomalies, new_predictions_binary)
print(f"New Precision {precision_new}")

# Calculation of F1-score
f1_new = f1_score(y_test_no_anomalies, new_predictions_binary)
print(f"New F1-Score: {f1_new}")

# Calculation of AUC-ROC
roc_auc_new = roc_auc_score(y_test_no_anomalies, new_predictions)
print(f"New ROC-AUC: {roc_auc_new}")

import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test_no_anomalies, new_predictions_binary)
roc_auc = auc(fpr, tpr)

# Calculation of the accuracy
accuracy_new = accuracy_score(y_test_no_anomalies, new_predictions_binary)
print(f"New Accuracy: {accuracy_new}")


plt.plot(fpr, tpr, label=f'AUC = {roc_auc_new}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[28]:


# Conversion to percent
accuracy_percent = accuracy * 100
accuracy_new_percent = accuracy_new * 100

# Plot the accuracy graph
plt.figure(figsize=(6, 4))
plt.plot(['Accuracy for poisoned data', 'Accuracy for cleaned data'], [accuracy_percent, accuracy_new_percent], marker='o', linestyle='-', color='b')
plt.title('Accuracy Comparison (with 17% of poisoned data)')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 100]) 
plt.grid(True)
plt.show()

# Dividing data into training and test sets
X_train_no_anomalies, X_test,  y_train_no_anomalies, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model on the new data set
history_new = model.fit(X_train_no_anomalies, y_train_no_anomalies, epochs=10, batch_size=32, validation_data=(X_test_no_anomalies, y_test_no_anomalies))
loss, accuracy = model.evaluate(X_test_no_anomalies, y_test_no_anomalies)
print(f"Loss: {loss}, accuracy: {accuracy}")

# Plotting accuracy over time for the new data set
plt.figure(figsize=(7, 4))
plt.plot(history_new.history['accuracy'], label='Training accuracy')
plt.plot(history_new.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy after removal of compromised nodes')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


# In[ ]:




