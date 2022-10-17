#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

d = pd.read_csv('train.csv').dropna(subset=['Age'])

d= d.drop(columns=["Name","Ticket","Fare","PassengerId","Parch","Embarked","Cabin"])
d["Sex"].replace(['female', 'male'],[0,1], inplace = True)
d.head()


# # MLPclassifier par défaut

# In[2]:


from sklearn.neural_network import MLPClassifier
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X= d[[ 'Pclass',  'Sex' , 'Age', 'SibSp']]
y= d['Survived' ]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0) 


start_time = time()
std_MLP1HL300 =  MLPClassifier(random_state=0).fit(X_train, y_train)
end_time = time()


ypred = std_MLP1HL300.predict(X_test)
score = accuracy_score(y_test, ypred)
print(f"Temps entraînement = {end_time - start_time}s")
print(f"je suis le score {score}")


# # MLP sans standardiser

# In[3]:



from sklearn.neural_network import MLPClassifier
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X= d[[ 'Pclass',  'Sex' , 'Age', 'SibSp']]
y= d['Survived' ]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0) 


start_time = time()
std_MLP1HL300 =  MLPClassifier( solver = 'adam',
                                learning_rate = 'constant',
                                hidden_layer_sizes=(500,400,300,200),
                                activation = 'identity', 
                                random_state=0).fit(X_train, y_train)
end_time = time()
ypred = std_MLP1HL300.predict(X_test)
score = accuracy_score(y_test, ypred)
print(f"Temps entraînement = {end_time - start_time}s")
print(f"je suis le score {score}")


# # MLP avec standardisation

# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X= d[[ 'Pclass',  'Sex' , 'Age', 'SibSp']]
y= d['Survived' ]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0) 

start_time = time()
std_MLP1HL300 = Pipeline([('scaler', StandardScaler()),
                          ('mlp', MLPClassifier( solver = 'adam',
                                                learning_rate = 'constant',
                                                hidden_layer_sizes=(500,400,300,200),
                                                activation = 'identity', 
                                                random_state=0))]
                         ).fit(X_train, y_train)
end_time = time()
ypred = std_MLP1HL300.predict(X_test)
score = accuracy_score(y_test, ypred)
print(f"Temps entraînement = {end_time - start_time}s")
print(f"je suis le score {score}")


# # Tensorflow

# In[56]:


scaler = StandardScaler()
df1 = pd.read_csv("train.csv").dropna(subset=["Age"])

df1.loc[df1["Sex"] == "male","Sex"] = 0
df1.loc[df1["Sex"] == "female","Sex"] = 1

X_train = df1.iloc[:, [2,4,5,6,7]].values
y_train = df1["Survived"].values
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation



classifier = Sequential()

# Couche cachée 500
classifier.add(Dense((500),input_shape=(5,)))
classifier.add(Activation('relu'))

# Couche cachée 400
classifier.add(Dense(400))
classifier.add(Activation('relu'))

# Couche cachée 300
classifier.add(Dense(300))
classifier.add(Activation('relu'))

# Couche de sortie 200
classifier.add(Dense(200))
classifier.add(Activation('relu'))

classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=2048, epochs=10,verbose=1)


classifier.evaluate(X_test,y_test)


# In[ ]:




