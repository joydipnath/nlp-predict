# -*- coding: utf-8 -*-
"""
    Questions - Project 1 - Sequential Models in NLP - Sentiment Classification.ipynb
    @author : Joydip Nath
    @Jyly 15th 2020
"""

"""
# from google.colab import drive
# drive.mount('/content/drive/')

# Commented out IPython magic to ensure Python compatibility.

"""
from tensorflow.keras.datasets import imdb
from keras.preprocessing import sequence
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

# Gensim contains word2vec models and processing tools


"""
# %matplotlib inline
"""


import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

"""
# path = '/content/drive/My Drive/Colab Notebooks/Natural Language Processing/'

# glove_file = datapath(path + 'glove.6B.50d.txt') # This is a GloVe model

"""

vocabulary_size = 10000
seq_length = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

"""### Pad each sentence to be of same length (2 Marks)
- Take maximum sequence length as 300
"""

X_train = sequence.pad_sequences(X_train, maxlen=seq_length)
X_test = sequence.pad_sequences(X_test, maxlen=seq_length)

"""### Print shape of features & labels (2 Marks)

Number of review, number of words in each review
"""

#### Add your code here ####
print("No of reviews:", X_train.shape)

#### Add your code here ####
length = [len(i) for i in X_train]
print("Average Review length:", np.mean(length))

"""Number of labels"""

#### Add your code here ####
print("No of labels:", X_test.shape)

"""### Print value of any one feature and it's label (2 Marks)

Feature value
"""

#### Add your code here ####
sample_id = 10
print(X_train[sample_id])

"""Label value"""

#### Add your code here ####
print("Label:", y_train[sample_id])

"""### Decode the feature value to get original sentence (2 Marks)

First, retrieve a dictionary that contains mapping of words to their index in the IMDB dataset
"""

#### Add your code here ####
index = imdb.get_word_index()

print(index)

"""Now use the dictionary to get the original words from the encodings, for a particular sentence"""

#### Add your code here ####
reverse_word_index = dict([(value, key) for (key, value) in index.items()])            
decoded_review = ' '.join([reverse_word_index.get(i - 3, "#") for i in X_train[0]])
print(decoded_review)

"""Get the sentiment for the above sentence
- positive (1)
- negative (0)
"""

#### Add your code here ####
print(y_train[0])

"""###
- Define a Sequential Model
- Add Embedding layer
  - Embedding layer turns positive integers into dense vectors of fixed size
  - `tensorflow.keras` embedding layer doesn't require us to onehot encode our words, instead we have to give each word a unique integer number as an id. For the imdb dataset we've loaded this has already been done, but if this wasn't the case we could use sklearn LabelEncoder.
  - Size of the vocabulary will be 10000
  - Give dimension of the dense embedding as 100
  - Length of input sequences should be 300
- Add LSTM layer
  - Pass value in `return_sequences` as True
- Add a `TimeDistributed` layer with 100 Dense neurons
- Add Flatten layer
- Add Dense layer
"""

#### Add your code here ####

from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
import os

def build_network(vocab_size, embedding_dim, sequence_length):
    input = Input(shape=(sequence_length,), name="Input")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length,
                          name="embedding")(input)
    lstm1 = LSTM(10, activation='tanh', return_sequences=False,
                 dropout=0.2, recurrent_dropout=0.2, name='lstm1')(embedding)
    output = Dense(1, activation='sigmoid', name='sigmoid')(lstm1)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_sentiment", name))
    # write_graph=True,write_grads=False
    
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True)
    
    return [tensorboard_callback, checkpoint_callback]

"""### Compile the model
- Use Optimizer as Adam
- Use Binary Crossentropy as loss
- Use Accuracy as metrics
"""

model = build_network(vocab_size=vocabulary_size, embedding_dim=100, sequence_length=seq_length)

callbacks = create_callbacks("sentiment")

# Commented out IPython magic to ensure Python compatibility.
#loading tensorboard extension
# %load_ext tensorboard.notebook
tensorboard = callbacks[0]
# %load_ext tensorboard
# %tensorboard_callback --logdir 
# log_dir = os.path.join(os.getcwd(), 'tb_log_sentiment', 'sentiment')
# log_dir

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /content/tb_log_sentiment/sentiment

"""### Print model summary (2 Marks)"""

#### Add your code here ####
model.summary()

"""### Fit the model (2 Marks)"""

#### Add your code here ####

result = model.fit(x=X_train, y=y_train,
              batch_size=32,
              epochs=10,
              validation_data=(X_test, y_test),
              callbacks=callbacks)

model.save("sentiment.h5")

"""### Evaluate model (2 Marks)"""

# loss_acc = model.evaluate(X_test, y_test, verbose=0)
# print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
#   (loss_acc[0], loss_acc[1]*100))

y_test_pred = model.predict(X_test)
y_test_pred = y_test_pred.reshape(y_test_pred.shape[0],)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

auc_lstm = roc_auc_score(y_test, y_test_pred)
auc_lstm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_pred = pd.DataFrame(data={'y_actual': y_test, 'y_pred': y_test_pred})

plt.figure(figsize=(16, 6))

msk = df_pred['y_actual'] == 0
sns.distplot(df_pred.loc[msk, 'y_pred'], label='negative reviews', kde=False)
sns.distplot(df_pred.loc[~msk, 'y_pred'], label='positive reviews', kde=False)
plt.legend()

"""### Predict on one sample (2 Marks)"""

print("New review: \'the movie was a great waste of my time\'")
review = "the movie was a great waste of my time"
words = review.split()
review = []
for word in words:
  if word not in index: 
    review.append(2)
  else:
    review.append(index[word]+3)
review = sequence.pad_sequences([review],
    truncating='pre',  padding='pre', maxlen=seq_length)

prediction = model.predict(review)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])

print("New review: \'the movie was a great use of my time\'")
review = "the movie was a great use of my time"
words = review.split()
review = []
for word in words:
  if word not in index: 
    review.append(2)
  else:
    review.append(index[word]+3)
review = sequence.pad_sequences([review],
    truncating='pre',  padding='pre', maxlen=seq_length)

prediction = model.predict(review)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])