
##	IMPORTING DEPENDENCIES

import numpy as np
import math
import pandas as pd
import re
import time
from google.colab import drive
from bs4 import BeautifulSoup

try:
  %tensorflow_version 2.x
except:
  pass
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds

print(tf.__version__)

## READING THE INPUT CSV USING PANDAS

cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv(
    "/content/drive/My Drive/CNN_for_NLP/data/training.csv", # PATH OF THE TRAINING DATASET
    header=None,
    names=cols, # READING DATA BELONGING TO THE COLUMN NAMES GIVEN IN cols[]
    engine="python",
    encoding="latin1"
)
test_data = pd.read_csv(
    "/content/drive/My Drive/CNN_for_NLP/data/testing.csv", # PATH OF THE TESTING DATASET
    header=None,
    names=cols, # READING DATA BELONGING TO THE COLUMN NAMES GIVEN IN cols[]
    engine="python",
    encoding="latin1"
)

train_data.head(5)
train_data.shape

## CLEANING OF DATASET

train_data.drop(["id", "date", "query","user"],axis=1,inplace=True) # DROPPING THE UNWANTED COLUMNS , THAT DONT CONTRIBUTE
# MAKING USE OF REGULAR EXPRESSION'S (RegEx)
def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet) # REPLACING TEXT BEGINNING WITH '@' WITH white space
    tweet = re.sub(r"https?://[A-Za-z0-0./]+", ' ', tweet) # REPLACING LINKS/URL's WITH white spaces
    tweet = re.sub(r"[^A-Za-z.?!']", ' ', tweet)  # REPLACING ANYTHING OTHER THAN SPECIFIED WITH white spaces
    tweet = re.sub(r" +", ' ',tweet) # REPLACING MULTIPLE white spaces WITH ONE white space
    return tweet

data_clean = [clean_tweet(tweet) for tweet in train_data.text] # CALLING THE clean_tweet() FUNCTION FOR ALL TWEETS FROM THE INPUT DATA
data_labels = train_data.sentiment.values
data_labels[data_labels == 4] = 1 # RESTRICTING THE VALUES BETWEEN 0 AND 1
set(data_labels)


## TOKENIZATION

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    data_clean,target_vocab_size=2**16
)
data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]


## PADDING

MAX_LEN = max(len(sentence) for sentence in data_inputs)
data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=MAX_LEN) # ADDING 0's AT THE END OF THE TEXT IF THE LENGTH OF THE TEXT 
																			# IS LESSER THAN THE MAX_LEN

## TRAIN / TEST SPLITTING 

test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.concatenate((test_idx,test_idx+800000))

test_inputs = data_inputs[test_idx] 
test_labels = data_labels[test_idx]
train_inputs = np.delete(data_inputs, test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)

## MODEL BUILDING

class DCNN(tf.keras.Model):
    def __init__(self,
                 vocab_size,  emb_dim=128,
                 nb_filters=50, FFN_units=512,
                 nb_classes=2, dropout_rate=0.1,
                 training=False, name="dcnn"):
        
        super(DCNN, self).__init__(name=name)

        self.embedding = layers.Embedding(vocab_size, emb_dim)
        # Layer 1
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2,
                                    padding="valid", activation="relu")
        self.pool_1 = layers.GlobalMaxPool1D()
       # Layer 2
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3,
                                    padding="valid", activation="relu")
        self.pool_2 = layers.GlobalMaxPool1D()
        # Layer 3
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4,
                                    padding="valid", activation="relu")
        self.pool_3 = layers.GlobalMaxPool1D()
        # Dense Fully Connected Layer
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        # Output Layer
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool_1(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool_2(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool_3(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1)   # (batchsize, 3*nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged)
        output = self.last_dense(merged)

        return output

## PARAMETER CONFIGURATION

VOCAB_SIZE = tokenizer.vocab_size
EMB_DIM = 64
NB_FILTERS = 50
FFN_UNITS = 128
NB_CLASSES = len(set(train_labels))
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
NB_EPOCHS = 2

## TRAINING

Dcnn = DCNN(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS, FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES, dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam", 
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

checkpoint_path = "./drive/My Drive/CNN_for_NLP/ckpt/"
ckpt = tf.train.Checkpoint(Dcnn=Dcnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored .")

Dcnn.fit(train_inputs,
         train_labels,
         batch_size = BATCH_SIZE,
         epochs=NB_EPOCHS)
ckpt_manager.save

## EVALUATION - TESTING

results = Dcnn.evaluate(test_inputs,
                        test_labels,
                        batch_size=BATCH_SIZE)
print(results)

Dcnn(np.array([tokenizer.encode("I hate you")]), training=False).numpy()