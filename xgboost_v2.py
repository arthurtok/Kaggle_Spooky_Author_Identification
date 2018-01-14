import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes

# Imports
import pandas as pd
import sys
import glob
import errno
import csv
import numpy as np
from nltk.corpus import stopwords
import re
import nltk.data
import nltk
import os
from collections import OrderedDict
from subprocess import check_call
from shutil import copyfile
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
# import mpld3
import seaborn as sns
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalAveragePooling1D,Merge,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras import backend as K
from sklearn.linear_model import SGDClassifier as sgd
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None

## Read the train and test dataset and check the top few lines ##
X_train = pd.read_csv( "../Data/train.csv", header=0,delimiter="," )
X_test = pd.read_csv( "../Data/test.csv", header=0,delimiter="," )
test_id = X_test['id'].values
wv = "../Data/glove.6B.100d.txt"
print("Number of rows in train dataset : ",X_train.shape[0])
print("Number of rows in test dataset : ",X_test.shape[0])
authors = ['EAP','MWS','HPL']
Y_train = LabelEncoder().fit_transform(X_train['author'])

# Clean data
def clean(X_train,X_test):
    X_train['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_train['text']]
    X_test['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_test['text']]
    return X_train,X_test
X_train,X_test = clean(X_train,X_test)

########################################
# FEATURE ENGINEERING
########################################
# Punctuation
punctuations = [{"id":1,"p":"[;:]"},{"id":2,"p":"[,.]"},{"id":3,"p":"[?]"},{"id":4,"p":"[\']"},{"id":5,"p":"[\"]"},{"id":6,"p":"[;:,.?\'\"]"}]
for p in punctuations:
    punctuation = p["p"]
    _train =  [ sentence.split() for sentence in X_train['text'] ]
    X_train['punc_'+str(p["id"])] = [len([word for word in sentence if bool(re.search(punctuation, word))])*100.0/len(sentence) for sentence in _train]    

    _test =  [ sentence.split() for sentence in X_test['text'] ]
    X_test['punc_'+str(p["id"])] = [len([word for word in sentence if bool(re.search(punctuation, word))])*100.0/len(sentence) for sentence in _test]    


# 1. META FEATURES
## Number of words in the text ##
X_train["num_words"] = X_train["text"].apply(lambda x: len(str(x).split()))
X_test["num_words"] = X_test["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
X_train["num_unique_words"] = X_train["text"].apply(lambda x: len(set(str(x).split())))
X_test["num_unique_words"] = X_test["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
X_train["num_chars"] = X_train["text"].apply(lambda x: len(str(x)))
X_test["num_chars"] = X_test["text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
X_train["num_stopwords"] = X_train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
X_test["num_stopwords"] = X_test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
X_train["num_punctuations"] =X_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
X_test["num_punctuations"] =X_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
X_train["num_words_upper"] = X_train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
X_test["num_words_upper"] = X_test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
X_train["num_words_title"] = X_train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
X_test["num_words_title"] = X_test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
X_train["mean_word_len"] = X_train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
X_test["mean_word_len"] = X_test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# 2. TEXT BASED FEATURES
# Feature Engineering
# tfidf - words - nb
def tfidfWords(X_train,X_test):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())
    return train_tfidf,test_tfidf,full_tfidf
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_tfidf_MNB(X_train,X_test,Y_train):
    train_tfidf,test_tfidf,full_tfidf = tfidfWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_tfidf_MNB(X_train,X_test,Y_train)
X_train["tfidf_words_nb_eap"] = pred_train[:,0]
X_train["tfidf_words_nb_hpl"] = pred_train[:,1]
X_train["tfidf_words_nb_mws"] = pred_train[:,2]
X_test["tfidf_words_nb_eap"] = pred_test[:,0]
X_test["tfidf_words_nb_hpl"] = pred_test[:,1]
X_test["tfidf_words_nb_mws"] = pred_test[:,2]

# tfidf - chars - nb
def tfidfWords(X_train,X_test):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,5),analyzer='char')
    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())
    return train_tfidf,test_tfidf
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do(X_train,X_test,Y_train):
    train_tfidf,test_tfidf = tfidfWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test
pred_train,pred_test = do(X_train,X_test,Y_train)
X_train["tfidf_chars_nb_eap"] = pred_train[:,0]
X_train["tfidf_chars_nb_hpl"] = pred_train[:,1]
X_train["tfidf_chars_nb_mws"] = pred_train[:,2]
X_test["tfidf_chars_nb_eap"] = pred_test[:,0]
X_test["tfidf_chars_nb_hpl"] = pred_test[:,1]
X_test["tfidf_chars_nb_mws"] = pred_test[:,2]

# Feature Engineering
# count - words - nb
def countWords(X_train,X_test):
    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_count = count_vec.transform(X_train['text'].values.tolist())
    test_count = count_vec.transform(X_test['text'].values.tolist())
    return train_count,test_count
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_count_MNB(X_train,X_test,Y_train):
    train_count,test_count=countWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_count[dev_index], train_count[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_count_MNB(X_train,X_test,Y_train)
X_train["count_words_nb_eap"] = pred_train[:,0]
X_train["count_words_nb_hpl"] = pred_train[:,1]
X_train["count_words_nb_mws"] = pred_train[:,2]
X_test["count_words_nb_eap"] = pred_test[:,0]
X_test["count_words_nb_hpl"] = pred_test[:,1]
X_test["count_words_nb_mws"] = pred_test[:,2]

# Feature Engineering
# count - chars - nb
def countChars(X_train,X_test):
    count_vec = CountVectorizer(ngram_range=(1,7),analyzer='char')
    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_count = count_vec.transform(X_train['text'].values.tolist())
    test_count = count_vec.transform(X_test['text'].values.tolist())
    return train_count,test_count
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_count_chars_MNB(X_train,X_test,Y_train):
    train_count,test_count=countChars(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_count[dev_index], train_count[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_count_chars_MNB(X_train,X_test,Y_train)
X_train["count_chars_nb_eap"] = pred_train[:,0]
X_train["count_chars_nb_hpl"] = pred_train[:,1]
X_train["count_chars_nb_mws"] = pred_train[:,2]
X_test["count_chars_nb_eap"] = pred_test[:,0]
X_test["count_chars_nb_hpl"] = pred_test[:,1]
X_test["count_chars_nb_mws"] = pred_test[:,2]

# load the GloVe vectors in a dictionary:

# def loadWordVecs():
#     embeddings_index = {}
#     f = open(wv)
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     print('Found %s word vectors.' % len(embeddings_index))
#     return embeddings_index

# def sent2vec(embeddings_index,s): # this function creates a normalized vector for the whole sentence
#     words = str(s).lower()
#     words = word_tokenize(words)
#     words = [w for w in words if not w in stopwords.words('english')]
#     words = [w for w in words if w.isalpha()]
#     M = []
#     for w in words:
#         try:
#             M.append(embeddings_index[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis=0)
#     if type(v) != np.ndarray:
#         return np.zeros(100)
#     return v / np.sqrt((v ** 2).sum())

# def doGlove(x_train,x_test):
#     embeddings_index = loadWordVecs()
#     # create sentence vectors using the above function for training and validation set
#     xtrain_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_train)]
#     xtest_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_test)]
#     xtrain_glove = np.array(xtrain_glove)
#     xtest_glove = np.array(xtest_glove)
#     return xtrain_glove,xtest_glove,embeddings_index

# glove_vecs_train,glove_vecs_test,embeddings_index = doGlove(X_train['text'],X_test['text'])
# X_train[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_train.tolist())
# X_test[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_test.tolist())


# Using Neural Networks and Facebook's Fasttext
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# NN
def doAddNN(X_train,X_test,pred_train,pred_test):
    X_train["nn_eap"] = pred_train[:,0]
    X_train["nn_hpl"] = pred_train[:,1]
    X_train["nn_mws"] = pred_train[:,2]
    X_test["nn_eap"] = pred_test[:,0]
    X_test["nn_hpl"] = pred_test[:,1]
    X_test["nn_mws"] = pred_test[:,2]
    return X_train,X_test

def initNN(nb_words_cnt,max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt,32,input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,
                     5,
                     padding='valid',
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

def doNN(X_train,X_test,Y_train):
    max_len = 70
    nb_words = 10000
    
    print('Processing text dataset')
    texts_1 = []
    for text in X_train['text']:
        texts_1.append(text)

    print('Found %s texts.' % len(texts_1))
    test_texts_1 = []
    for text in X_test['text']:
        test_texts_1.append(text)
    print('Found %s texts.' % len(test_texts_1))
    
    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(texts_1 + test_texts_1)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
    del test_sequences_1
    del sequences_1
    nb_words_cnt = min(nb_words, len(word_index)) + 1

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN(nb_words_cnt,max_len)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=4, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_pad)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN(X_train,X_test,pred_train,pred_full_test/5)

## NN Glove

def doAddNN_glove(X_train,X_test,pred_train,pred_test):
    X_train["nn_glove_eap"] = pred_train[:,0]
    X_train["nn_glove_hpl"] = pred_train[:,1]
    X_train["nn_glove_mws"] = pred_train[:,2]
    X_test["nn_glove_eap"] = pred_test[:,0]
    X_test["nn_glove_hpl"] = pred_test[:,1]
    X_test["nn_glove_mws"] = pred_test[:,2]
    return X_train,X_test

def initNN_glove():
    # create a simple 3 layer sequential neural net
    model = Sequential()

    model.add(Dense(128, input_dim=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# def doNN_glove(X_train,X_test,Y_train,xtrain_glove,xtest_glove):
#     # scale the data before any neural net:
#     scl = preprocessing.StandardScaler()
#     ytrain_enc = np_utils.to_categorical(Y_train)
#     kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
#     cv_scores = []
#     pred_full_test = 0
#     xtrain_glove = scl.fit_transform(xtrain_glove)
#     xtest_glove = scl.fit_transform(xtest_glove)
#     pred_train = np.zeros([xtrain_glove.shape[0], 3])
    
#     for dev_index, val_index in kf.split(xtrain_glove):
#         dev_X, val_X = xtrain_glove[dev_index], xtrain_glove[val_index]
#         dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
#         model = initNN_glove()
#         model.fit(dev_X, y=dev_y, batch_size=32, epochs=10, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
#         pred_val_y = model.predict(val_X)
#         pred_test_y = model.predict(xtest_glove)
#         pred_full_test = pred_full_test + pred_test_y
#         pred_train[val_index,:] = pred_val_y
#     return doAddNN_glove(X_train,X_test,pred_train,pred_full_test/5)

# Fast Text

def doAddFastText(X_train,X_test,pred_train,pred_test):
    X_train["ff_eap"] = pred_train[:,0]
    X_train["ff_hpl"] = pred_train[:,1]
    X_train["ff_mws"] = pred_train[:,2]
    X_test["ff_eap"] = pred_test[:,0]
    X_test["ff_hpl"] = pred_test[:,1]
    X_test["ff_mws"] = pred_test[:,2]
    return X_train,X_test


def initFastText(embedding_dims,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def preprocessFastText(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs

def doFastText(X_train,X_test,Y_train):
    min_count = 2

    docs = create_docs(X_train)
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    docs = tokenizer.texts_to_sequences(docs)

    maxlen = 300

    docs = pad_sequences(sequences=docs, maxlen=maxlen)
    input_dim = np.max(docs) + 1
    embedding_dims = 20

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)

    docs_test = create_docs(X_test)
    docs_test = tokenizer.texts_to_sequences(docs_test)
    docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)
    xtrain_pad = docs
    xtest_pad = docs_test
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initFastText(embedding_dims,input_dim)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=25, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(docs_test)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddFastText(X_train,X_test,pred_train,pred_full_test/5)

X_train,X_test = doFastText(X_train,X_test,Y_train)
X_train,X_test = doNN(X_train,X_test,Y_train)
# X_train,X_test = doNN_glove(X_train,X_test,Y_train,glove_vecs_train,glove_vecs_test)

# Final model
# Final Model
# XGBoost
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.11
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 3000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

n_splits = 20

def do(X_train,X_test,Y_train):
    drop_columns=["id","text","words"]
    x_train = X_train.drop(drop_columns+['author'],axis=1)
    x_test = X_test.drop(drop_columns,axis=1)
    y_train = Y_train
    
    kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([x_train.shape[0], 3])
    for dev_index, val_index in kf.split(x_train):
        dev_X, val_X = x_train.loc[dev_index], x_train.loc[val_index]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, x_test, seed_val=0, colsample=0.7)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("cv scores : ", cv_scores)
    return pred_full_test/n_splits
result = do(X_train,X_test,Y_train)

out_df = pd.DataFrame(result)
out_df.columns = ['EAP', 'HPL', 'MWS']
out_df.insert(0, 'id', test_id)
out_df.to_csv("sub_v3.csv", index=False)
