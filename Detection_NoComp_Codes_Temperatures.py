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


import random


# In[22]:


indices_nodes = random.sample(range(len(df)), number_nodes)


# In[23]:


selected_nodes = df.iloc[indices_nodes]


# In[24]:


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", selected_nodes)


# In[25]:


#Update the value in the csv file using the replace() function
# updating the column value/data
for i in df.iloc[indices_nodes]:
    
    df.loc[indices_nodes, 'temp'] = df.loc[indices_nodes, 'temp'].replace({29: 15})

# writing into the file
df.to_csv("IOT_temp.csv", index=False)
print(df)


# In[28]:


from sklearn.model_selection import train_test_split

# Select relevant columns
data_selected = df[["day","month", "time", "temp", "location"]]

seuil_max = ""  # set maximum acceptable temperature
seuil_min = ""  # set the maximum acceptable temperature

# Add an "anomaly" column for temperatures outside the range
data_selected['anomaly'] = data_selected['temp'].apply(lambda x: 0 if x < seuil_min or x > seuil_max else 1)

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


# In[29]:


# Tracing accuracy through the ages
plt.figure(figsize=(7, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and validation Accuracy with compromised nodes')
plt.xlabel('Epoch')
plt.ylabel(f"accuracy: {accuracy*100}")
plt.legend()
plt.show()


# In[30]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, accuracy: {accuracy}")


# In[31]:


from sklearn.metrics import accuracy_score
# Calculating accuracy
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy}")


# In[32]:


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



# In[33]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[34]:


# Test set prediction
predictions = model.predict(X_test)
# Calculate the F1 score
f1 = f1_score(y_test, predictions_binary)
print(f1)


# In[35]:


from sklearn.metrics import precision_score

precision = precision_score(y_test, predictions_binary)
print(precision)


# In[36]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, predictions_binary)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[37]:


# Obtaining anomaly indexes
anomaly_indices = [i for i in range(len(predictions_binary)) if predictions_binary[i] == 0]

# Delete the lines corresponding to anomalies in your test set
X_test_no_anomalies = np.delete(X_test, anomaly_indices, axis=0)
y_test_no_anomalies = np.delete(y_test, anomaly_indices, axis=0)


# In[38]:


# Creating of a new pandas DataFrame
df_no_anomalies = pd.DataFrame(X_test_no_anomalies, columns=['anomaly'])
df_no_anomalies['cleandata'] = y_test_no_anomalies


# In[39]:


from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc,accuracy_score
import matplotlib.pyplot as plt
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


# In[41]:


# Calculation of the accuracy
accuracy_new = accuracy_score(y_test_no_anomalies, new_predictions_binary)
print(f"New Accuracy: {accuracy_new}")

# Plot the accuracy graph
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy for poisoned data', 'Accuracy for cleaned data'], [accuracy, accuracy_new], color=['red', 'green'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()


# In[42]:


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


# In[43]:


# Dividing data into training and test sets
X_train_no_anomalies, X_test,  y_train_no_anomalies, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model on the new data set
history_new = model.fit(X_train_no_anomalies, y_train_no_anomalies, epochs=10, batch_size=32, validation_data=(X_test_no_anomalies, y_test_no_anomalies))
loss, accuracy = model.evaluate(X_test_no_anomalies, y_test_no_anomalies)
print(f"Loss: {loss}, accuracy: {accuracy}")


# In[44]:


# Plotting accuracy over time for the new data set
plt.figure(figsize=(7, 4))
plt.plot(history_new.history['accuracy'], label='Training accuracy')
plt.plot(history_new.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy after removal of compromised nodes')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

