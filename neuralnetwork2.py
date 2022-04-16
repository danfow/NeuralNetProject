from cgi import test
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
tweets = pd.read_csv("congressional_tweet_training_data.csv")






tweets.head()
tweets.pop("full_text")

tweets['party_id'] = np.where(tweets['party_id']=='D',0,1)

top_unique_hashtags = tweets['hashtags'].value_counts().index.to_list()[:500]

temp2 = tweets.loc[~tweets['hashtags'].isin(top_unique_hashtags)]
temp2 = list(temp2.index)
#print (temp)
for index2 in temp2:
  tweets.at[index2,'hashtags'] = 'other'


tweets['year'] = tweets['year'].astype(str)

#tweets, testing_data = train_test_split(alldata, test_size=0.2, random_state=25)
##print(f"No. of training examples: {tweets.shape[0]}")
#print(f"No. of testing examples: {testing_data.shape[0]}")
#test_labels = testing_data.pop('party_id')



#training_data, testing_data = train_test_split(tweets, test_size=0.2, shuffle=False,stratify=None)

training_data, testing_data = train_test_split(tweets, test_size=0.2)
print(f"No. of training examples: {tweets.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


#tweet_features = tweets.copy()

tweet_features = training_data.copy()

tweet_features2 = testing_data.copy() #2


print(tweets.head)


tweet_labels = tweet_features.pop('party_id')
test_labels = tweet_features2.pop('party_id') #2

tweet_labels = tweet_labels.to_numpy() #try i guess
test_labels = test_labels.to_numpy() #try i guess

print("labels")
print(test_labels)

#tweet_features['year'] = tweet_features['year'].astype(str)



inputs = {}
for name, column in tweet_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


print (inputs)

inputs2 = {}
for name, column in tweet_features2.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs2[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype) #2






numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
#norm.adapt(np.array(tweets[numeric_inputs.keys()]))
norm.adapt(np.array(training_data[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

preprocessed_inputs = [all_numeric_inputs]







numeric_inputs2 = {name:input for name,input in inputs2.items()
                  if input.dtype==tf.float32}

x2 = layers.Concatenate()(list(numeric_inputs2.values()))
norm2 = layers.Normalization()
norm2.adapt(np.array(testing_data[numeric_inputs2.keys()]))
all_numeric_inputs2 = norm2(x)



preprocessed_inputs2 = [all_numeric_inputs2]






for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(tweet_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(),output_mode = "multi_hot")

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
tweet_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)




for name, input in inputs2.items():
  if input.dtype == tf.float32:
    continue

  lookup2 = layers.StringLookup(vocabulary=np.unique(tweet_features2[name]))
  one_hot2 = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(),output_mode = "multi_hot")

  x = lookup2(input)
  x = one_hot2(x)
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







def tweet_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    #layers.Dense(input_shape = (516,)),
    layers.Dense(50),
    layers.Dense(500),
    layers.Dense(50),
    layers.Dense(50),
    layers.Dense(150,activation = 'relu'),
    layers.Dense(2),
    #layers.Softmax()
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  #model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                #optimizer=tf.optimizers.Adam(),
                #metrics=['accuracy'])
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model


#print(tweets.dtypes)
#print("hello")
#print(tweet_features.dtypes)

tweet_models = tweet_model(tweet_preprocessing, inputs)


print (tweet_features.info())

#splitting data



tweet_models.fit(x=tweet_features_dict, y=tweet_labels, epochs=3)

test_loss, test_acc = tweet_models.evaluate(tweet_features_dict2,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

print("tweet model ",tweet_models )

#probability_model = tf.keras.Sequential([tweet_models, 
                                         #tf.keras.layers.Softmax()])

predictions = tweet_models.predict(tweet_features_dict2)


print(predictions)
for i in range (0,100):
  #print ("prediction of neural net is ", np.argmax(predictions[i]), "real value is ", test_labels.iloc[i])
  print ("prediction values were ", predictions[i],"prediction of neural net is ", np.argmax(predictions[i]), "real value is ", test_labels[i])
  #print("real value is" , test_labels.iloc[i])



#for prediction in range (1,101):
  #print ("prediction of neural net is ", np.argmax(predictions[prediction]))
  #print (" real value is ",test_labels[prediction])

#for index,value in test_labels.items():
   #print ("prediction of neural net is ", np.argmax(predictions[index]), "real value is ", value)
   #print("real value is" , value)


