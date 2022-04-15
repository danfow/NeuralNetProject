# TensorFlow and tf.keras
from enum import unique
from turtle import shape
from xml.etree.ElementTree import tostring
from certifi import where
from requests import head
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # manipulate data sets
import logging

dftrain = pd.read_csv("congressional_tweet_training_data.csv")
dfeval  = pd.read_csv("validate.csv")
y_train = dftrain.pop('party_id')
y_eval = dfeval.pop('party_id')

dftrain.pop("full_text")
dfeval.pop("full_text")
print(dftrain.head(20))





dftrain['hashtags'] = dftrain['hashtags'].str.lower()
dfeval['hashtags'] = dfeval['hashtags'].str.lower()

#top_unique_hashtags = dftrain['hashtags'].value_counts().index.to_list()[:50] #gets the top 50 hashtags
print (dftrain['hashtags'].value_counts())
#top_unique_hashtags = dftrain['hashtags'].value_counts().index.to_list()[:10] #gets the top 500 hashtags

top_unique_hashtags = dftrain['hashtags'].value_counts().index.to_list()[:500] #gets the top 500 hashtags
top_unique_hashtags2 = dfeval['hashtags'].value_counts().index.to_list()[:500] #gets the top 500 hashtags

print("top 500 unique hashtags are",top_unique_hashtags)

all_hash_tags = dftrain['hashtags'].unique().tolist()
#print (all_hash_tags)
print ("total unique hashtags is" + str(len(all_hash_tags)))


#print  (all_hash_tags[:10])
#k = 0
#for i in all_hash_tags:
  #k += 1
  #if (k % 10000 == 0):
    #print(k,"iterations")
    #print (dftrain.head())
  #if (i not in top_unique_hashtags):
    #dftrain['hashtags'].replace(i,"other",inplace=True)
    #dftrain['hashtags'][i] = "other"
    #dftrain.at[i,"hashtag"] = "other"



#dftrain['hashtags'] = dftrain.loc[~dftrain['hashtags'].isin(top_unique_hashtags)]["hashtags"] = "other" so close?


#next five lines of spaghetti code will convert all hashtags that are not a part of the top hastags to 'other'

temp = dftrain.loc[~dftrain['hashtags'].isin(top_unique_hashtags)]
temp = list(temp.index)
#print (temp)
for index in temp:
  dftrain.at[index,'hashtags'] = 'other'


temp2 = dfeval.loc[~dfeval['hashtags'].isin(top_unique_hashtags2)]
temp2 = list(temp2.index)
#print (temp)
for index2 in temp2:
  dfeval.at[index2,'hashtags'] = 'other'






print(dftrain.head(20))

CATEGORICAL_COLUMNS = ['hashtags'],['year']
NUMERIC_COLUMNS = ['favorite_count', 'retweet_count']

#feature_columns = [] #feed to linear model to make predictions
#for feature_name in CATEGORICAL_COLUMNS:
  #vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  #feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

#for feature_name in NUMERIC_COLUMNS:
  #feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
 
dftrain.to_csv("testingData.csv", index=False)
dfeval.to_csv("validationSet.csv",index=False)

#print(feature_columns)



#print (dftrain['hashtags'].unique())



#def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  #def input_function():  # inner function, this will be returned
    #ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    #if shuffle:
     # ds = ds.shuffle(1000)  # randomize order of data
    #ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    #return ds  # return a batch of the dataset
  #return input_function  # return a function object for use


#train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
#eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


#model = keras.Sequential([
    #keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    #keras.layers.Dense(2, activation='softmax') # output layer (3)
#])


#model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              #metrics=['accuracy'])


#print(dftrain.head())
#model.fit(dftrain, y_train, epochs=10)  # we pass the data, labels and epochs and watch the magic!