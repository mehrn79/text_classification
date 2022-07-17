---
jupyter:
  accelerator: GPU
  colab:
    name: train_model\_&\_predict.ipynb
  gpuClass: standard
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

# **Text classification model and predict**

``` {.python}
import json
import nltk
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import collections
import re
from collections import Counter, defaultdict
from oauth2client.client import GoogleCredentials
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Convolution1D
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras import optimizers
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import codecs
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
```
## **خواندن دیتا**
``` {.python}
! gdown --id 1wbZQm1FimgMmRDUXCnLa1wBRz7DnJZJd
! gdown --id 1vaX5SO_wGoiucO9P-bF--J7WNj-qUJhx
```

``` {.python}
with open('/content/chart.csv') as f:
  chart = f.read()
```

``` {.python}
chart = pd.read_csv('chart.csv')
text = chart['clean_text'].values
Categories= chart['Cat'].values
```
## **ساختن مدل شبکه عصبی** 
در این مدل از شبکه عصبی به عنوان x_train متن خبر ها و به عنوان y_train دسته بندی ها را قرار می دهیم
``` {.python}
category_le = LabelEncoder()
Categories = category_le.fit_transform(Categories)

category_x = text

category_y = Categories
```
20 درصد  دیتا را به عنوان تست و اعتبار سنجی به شبکه می دهیم
``` {.python}
x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(category_x, category_y, test_size=0.20)
print('--Category Predictor--')
print('Train size: ',len(x_cat_train))
print('Test size : ',len(x_cat_test))
```
 در این بخش به توکنایز کردن متن خبر ها یعنی تبدیل متن به عدد جهت استفاده در شبکه خصبی می پردازیم
``` {.python}
cat_vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=0.005)
# train set
x_cat_train_corpus = x_cat_train
# test set
x_cat_test_corpus = x_cat_test

# Fit vectorizer
cat_vectorizer.fit(x_cat_train_corpus)

# Print vocabulary
cat_vocab = cat_vectorizer.get_feature_names()
print(cat_vocab)
print('Vocab length:', len(cat_vocab))

# Transform vectorizer over train and test set
x_cat_train_vec = cat_vectorizer.transform(x_cat_train_corpus).toarray()
x_cat_test_vec = cat_vectorizer.transform(x_cat_test_corpus).toarray()
```
در این قسمت باتوجه به این که اکثر اخبار متنی در حدود 500 کلمه دارند برای کاهش پارامترهای شبکه عصبی تمامی متن ها را به 500 کلمه محدود می کنیم
``` {.python}
titles=[]
test_titles=[]

for data in x_cat_train :
  arr=data.split(' ')
  s=' '.join(arr[:500])
  titles.append(s)

for data in x_cat_test :
  arr=data.split(' ')
  s=' '.join(arr[:500])
  test_titles.append(s)
```
مقدار دهی lstm_x_train و lstm_x_test بر اساس دیتا محدود شده
``` {.python}
lstm_x_train = titles

# test set
lstm_x_test = test_titles
```

``` {.python}
num_words = 10000

# create the tokenizer
tokenizer = Tokenizer(num_words=num_words)

# fit the tokenizer on the documents
tokenizer.fit_on_texts(x_cat_train_corpus)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(lstm_x_train)

# pad sequences
max_length = max([len(s.split()) for s in lstm_x_train])
x_train_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(lstm_x_test)
x_test_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
```
ساختن بخش y_train با تبدیل دسته بندی ها به one hot
``` {.python}
y_cat_train_classes = np.unique(y_cat_train)
y_cat_train_classes_len = len(y_cat_train_classes)

y_cat_test_classes = np.unique(y_cat_train)
y_cat_test_classes_len = len(y_cat_test_classes)
```
ساخت مدل شبکه عصبی و لایه های آن و ذخیره اطلاعات بهترین epoch از نظر دقت
``` {.python}
categorical_y_train = to_categorical(y_cat_train, y_cat_train_classes_len)
categorical_y_test = to_categorical(y_cat_test, y_cat_test_classes_len)

model_name = "text_classification.h5"
mcp_save = ModelCheckpoint(filepath=model_name,
                           save_best_only=True,
                           monitor='val_loss',
                           mode='min',
                           verbose=1)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length))
model.add(Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(100, return_sequences=True, name='lstm_layer')))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(y_cat_train_classes_len, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[categorical_accuracy])

model.summary()
batch_size = 8
epochs = 10
hist = model.fit(x_train_padded, categorical_y_train, batch_size=batch_size, epochs=epochs,
                           callbacks=mcp_save)

loss, acc = model.evaluate(x_test_padded, categorical_y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))
model.save("final_epoch_"+model_name)
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 500, 50)           17322550  
                                                                     
     bidirectional_1 (Bidirectio  (None, 500, 200)         121600    
     nal)                                                            
                                                                     
     global_max_pooling1d_1 (Glo  (None, 200)              0         
     balMaxPooling1D)                                                
                                                                     
     dropout_2 (Dropout)         (None, 200)               0         
                                                                     
     dense_2 (Dense)             (None, 100)               20100     
                                                                     
     dropout_3 (Dropout)         (None, 100)               0         
                                                                     
     dense_3 (Dense)             (None, 105)               10605     
                                                                     
    =================================================================
    Total params: 17,474,855
    Trainable params: 17,474,855
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/10
    16523/16523 [==============================] - ETA: 0s - loss: 1.2603 - categorical_accuracy: 0.6680WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 575s 35ms/step - loss: 1.2603 - categorical_accuracy: 0.6680
    Epoch 2/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.8353 - categorical_accuracy: 0.7640WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 573s 35ms/step - loss: 0.8353 - categorical_accuracy: 0.7640
    Epoch 3/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.7191 - categorical_accuracy: 0.7912WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 577s 35ms/step - loss: 0.7191 - categorical_accuracy: 0.7912
    Epoch 4/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.6408 - categorical_accuracy: 0.8106WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 578s 35ms/step - loss: 0.6408 - categorical_accuracy: 0.8106
    Epoch 5/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.5750 - categorical_accuracy: 0.8285WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 578s 35ms/step - loss: 0.5750 - categorical_accuracy: 0.8285
    Epoch 6/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.5232 - categorical_accuracy: 0.8418WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 577s 35ms/step - loss: 0.5232 - categorical_accuracy: 0.8418
    Epoch 7/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.4736 - categorical_accuracy: 0.8552WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 577s 35ms/step - loss: 0.4736 - categorical_accuracy: 0.8552
    Epoch 8/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.4346 - categorical_accuracy: 0.8665WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 578s 35ms/step - loss: 0.4346 - categorical_accuracy: 0.8665
    Epoch 9/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.4025 - categorical_accuracy: 0.8743WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 578s 35ms/step - loss: 0.4025 - categorical_accuracy: 0.8743
    Epoch 10/10
    16523/16523 [==============================] - ETA: 0s - loss: 0.3732 - categorical_accuracy: 0.8828WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    16523/16523 [==============================] - 580s 35ms/step - loss: 0.3732 - categorical_accuracy: 0.8828
    Test Accuracy: 76.180965

در این بخش مدل train شده را برای استفاده های بعدی ذخیره در گوگل درایو ذخیره می کنیم
``` {.python}
from google.colab import drive
drive.mount('/content/drive')

!cp -r "/content/final_epoch_text_classification.h5" "/content/drive/MyDrive"
```

# **پیاده سازی بخش پیش بینی**

نصب و وارد کردن پیش نیازها
``` {.python}
pip install hazm
```

``` {.python}
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from hazm import *
import string
from string import digits
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
```
خواندن لیست stop words
``` {.python}
with open('/content/PersianStopWords.txt') as file:
  stopLines = file.readlines()
  stopWord = [item.replace('\n',"") for item in stopLines]
```
خواندن ستون دسته بندی از فایل csv
``` {.python}
Categories_type= chart['Cat'].values
```
حذف اعداد از متن
``` {.python}
def remove_digits(sent):
  remove_digits = str.maketrans('', '', digits)
  sentance =' '.join([w.translate(remove_digits) for w in sent.split(' ')]) 
  return sentance
```
تمیز کردن متن مورد نظر 
``` {.python}
def cleaning(sentance) :
  words= []
  cleanSent=[]
  stemmer = Stemmer()

  words=word_tokenize(sentance)

  for wordSent in words :
    if wordSent not in stopWord :
      stermmering=stemmer.stem(wordSent)
      cleanSent.append(stermmering)
  
  sent =' '.join(map(str, cleanSent))

  return remove_digits(sent)
```
تابع پیش بینی دسته بندی ها
``` {.python}
def predict_cat(texts):
  cats=[]
  clean_text=[]
  num_words = 10000
  # tokenizer = Tokenizer(num_words=num_words)
  # tokenizer.fit_on_texts(x_cat_train)

  for x in texts:
    clean_text.append(cleaning(x))
  
  for text in clean_text :
 
    text2sent = tokenizer.texts_to_sequences([text])
    
    max_length = max([len(s.split()) for s in text])
    pad_text2sent = pad_sequences(text2sent, maxlen=max_length)
    
    cat = model.predict([pad_text2sent])[0]
    cat_index = np.where(cat == max(cat))
    
    cats.append(Categories_type[cat_index])
  return cats
```
وارد کردن دیتا جهت پیش بینی دسته بندی
``` {.python}
text = chart['text'].values
```

``` {.python}
text=text[:100]
```

``` {.python}
arr=predict_cat(text)
```
``` {.python}
arr
```

::: {.output .execute_result execution_count="28"}
    [array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['kharj'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['eqtes'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['elmfa'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['shahr'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object),
     array(['havad'], dtype=object)]
:::
:::
