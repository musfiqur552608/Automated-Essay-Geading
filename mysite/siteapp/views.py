from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
import requests
import numpy as np
import pandas as pd
import os
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from docx import Document
import docx

#those files for model
path = os.path.join(os.path.dirname(__file__), 'final_lstm.h5')
path1 = os.path.join(os.path.dirname(__file__), 'word2vecmodel.bin')

#this url for accessing api
URL = "http://127.0.0.1:8004/essaygradeapi/"

# make sentence to word
def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x=re.sub("[^A-Za-z]"," ",x)
    x.lower()
    filtered_sentence = [] 
    words=x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence


#  make essay to word
def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sent2word(i))
    return final_words

# make vectorization
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index2word)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model[i])        
    vec = np.divide(vec,noOfWords)
    return vec

# get vectors
def getVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c+=1
    return essay_vecs

# get the model
def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model

#convert to vector
def convertToVec(text):
    content=text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format(path1, binary=True)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features )
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model(path)
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))

#this function convert the file data to text
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

#This is for homepage
def home(request):
    return render(request, 'mainpage.html')

#This is main function where we done everything for grading
def site(request):
    # try:
        if request.method == 'POST':
            val = request.POST['var']
            #this segment for working with file
            if request.FILES:
                K.clear_session()
                rawtext = request.FILES['filename']
                x = getText(rawtext)
                # grade here
                score = convertToVec(x)
                score1 = (float(score)/10.0)*float(val)
                mydict = {
                    "mytext" : x,
                    "score" : score1,
                    "out" : val,
                }
                json_data = json.dumps(mydict)
                # request api for post the data
                r = requests.post(url = URL, data=json_data)

                data = r.json()   
                K.clear_session()
            else:
                #this segment for working with text
                K.clear_session()
                text = request.POST['rawtext']
                # grade here
                score = convertToVec(text)
                score1 = (float(score)/10.0)*float(val)
                mydict = {
                    "mytext" : text,
                    "score" : score1,
                    "out" : val,
                }
                json_data = json.dumps(mydict)
                # request api for post the data
                r = requests.post(url = URL, data=json_data)

                data = r.json()   
                K.clear_session()	
        return render(request, 'mainpage.html', context=mydict)
    # except: 
    #     return HttpResponse("Opps...!\nSomething else went wrong.")


    
