from cgi import test
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#resource list used to build project (in order of importance)
#https://www.tensorflow.org/tutorials/load_data/csv
#https://www.tensorflow.org/tutorials/keras/classification
#https://www.tensorflow.org/guide/keras/train_and_evaluate
#https://youtu.be/tPYj3fFJGjk
#https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax


tweets = pd.read_csv("congressional_tweet_training_data.csv")
tweets.head()
tweets.pop("full_text") #not dealing with full text. Satisfied with the performance without it and network a little unweildy as is

tweets['party_id'] = np.where(tweets['party_id']=='D',0,1)  #converting all the R and D values to 0 or 1 to make tensorFlow happy
tweets['hashtags'] = tweets['hashtags'].str.lower() # converting all hashtags to lowercase to ensure hashtags that only differ in casing don't count as different hash tag types

top_unique_hashtags = tweets['hashtags'].value_counts().index.to_list()[:10000] #getting the top 10,000 hash tags. Performance of network seems to correlate pretty linearly with this but pretty performance intensive and worry of overfitting

temp2 = tweets.loc[~tweets['hashtags'].isin(top_unique_hashtags)]
temp2 = list(temp2.index)
for index2 in temp2:
  tweets.at[index2,'hashtags'] = 'other'


tweets['year'] = tweets['year'].astype(str) # converting years to string type because there are some irregular values and I don't want tensorFlow to freak out when treating them as numbers

training_data, testing_data = train_test_split(tweets, test_size=0.2) # randomly splits data into training and testing data with a 80/20 split. Validation set will be created from training data further down
print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")



tweet_features = training_data.copy()

tweet_features2 = testing_data.copy() 

print ("Example of new data format")
print(tweets.head)


tweet_labels = tweet_features.pop('party_id')
test_labels = tweet_features2.pop('party_id') 

tweet_labels = tweet_labels.to_numpy() 
test_labels = test_labels.to_numpy() 


#formatting magic adapted from tensorflow titanic data set examples https://www.tensorflow.org/tutorials/load_data/csv
inputs = {}
for name, column in tweet_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


inputs2 = {}
for name, column in tweet_features2.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs2[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype) #2


#normalizating numeric columns
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(training_data[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]



numeric_inputs2 = {name:input for name,input in inputs2.items()
                  if input.dtype==tf.float32}
x2 = layers.Concatenate()(list(numeric_inputs2.values()))
norm2 = layers.Normalization()
norm2.adapt(np.array(testing_data[numeric_inputs2.keys()]))
all_numeric_inputs2 = norm2(x)
preprocessed_inputs2 = [all_numeric_inputs2]

#more tensorflow magic taken from the documentation. Converting the non numeric categories to something tensorflow can build a model with. https://www.tensorflow.org/tutorials/load_data/csv
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(tweet_features[name]))
  multi_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(),output_mode = "multi_hot") #I think multi-hot is what I want here since tweets can have more than one hash tag?

  x = lookup(input)
  x = multi_hot(x)
  preprocessed_inputs.append(x)
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
tweet_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)


for name, input in inputs2.items():
  if input.dtype == tf.float32:
    continue

  lookup2 = layers.StringLookup(vocabulary=np.unique(tweet_features2[name]))
  multi_hot2 = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(),output_mode = "multi_hot")

  x = lookup2(input)
  x = multi_hot2(x)
  preprocessed_inputs2.append(x)
preprocessed_inputs_cat2 = layers.Concatenate()(preprocessed_inputs)
tweet_preprocessing2 = tf.keras.Model(inputs, preprocessed_inputs_cat)



tweet_features_dict = {name: np.array(value) 
                         for name, value in tweet_features.items()}

features_dict = {name:values[:1] for name, values in tweet_features_dict.items()}
tweet_preprocessing(features_dict)


tweet_features_dict2 = {name: np.array(value) 
                         for name, value in tweet_features2.items()}

features_dict2 = {name:values[:1] for name, values in tweet_features_dict2.items()}
tweet_preprocessing2(features_dict2)


#Adapated this model code for this dataset from more tensorflow docs https://www.tensorflow.org/tutorials/keras/classification

def tweet_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(50),
    layers.Dense(15,activation = 'relu'), #hopefully stops math from blowing up too much
    layers.Dense(2,activation=tf.keras.activations.softmax), #converting to a probability distribution for some nice output at the end
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)


  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # should be set to false since last layer is using a probability distribution not logits
              metrics=['accuracy'])
  return model


tweet_models = tweet_model(tweet_preprocessing, inputs)

tweet_models.fit(x=tweet_features_dict, y=tweet_labels,validation_split = 0.3, epochs=5) # taking 30% of training data and turning it into validation data

test_loss, test_acc = tweet_models.evaluate(tweet_features_dict2,  test_labels, verbose=2)

print('\n model accuracy on test data:', test_acc)

predictions = tweet_models.predict(tweet_features_dict2)

print("first 100 predictions")
for i in range (0,100):
  predicted_value = np.argmax(predictions[i]) # predictions are represented as a 2 element prob dist, highest value taken as our prediction. if 1st element (index 0) is picked model predicted democrat, if index 1 republican
  if (predicted_value == 1):
    predicted_value = 'R'
  else:
    predicted_value = 'D'
  
  real_value = test_labels[i] # need to compare to real values

  if (real_value == 1):
    real_value = 'R'
  else:
    real_value = 'D'

  print ("prediction probabilites were ", predictions[i],"prediction of neural net is ", predicted_value, "real value is ", real_value)



