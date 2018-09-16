import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import re, collections
from sklearn.ensemble import RandomForestRegressor
from itertools import chain
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#df = pd.read_excel("/Users/pengluyao/.kaggle/training_set_rel3.xls")

df = pd.read_excel("training_set_rel3.xls")

names = list(df.columns.values)
data = df[['essay_set','essay','domain1_score']].copy()
print(data)

df.set_index('essay_id')
df.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))

#get the essays and essay_set into dictionary
train_data = {}
for index, row in df.iterrows():
    essay = row["essay"].strip().split(" ")
    essay_set = row['essay_set']
    domain1_score = row['domain1_score']/2
    length = len(essay)
    if essay_set not in train_data:
        train_data[essay_set] = {"essay":[], "score":[], "length":[]}
    train_data[essay_set]["essay"].append(essay)
    train_data[essay_set]['score'].append(domain1_score)
    train_data[essay_set]['length'].append(length)

total_len = []
for set in range(1,9):
    len_by_set = train_data[set]["length"]
    total_len.append(len_by_set)

total_len = list(chain(*total_len))
df['total_len'] = pd.Series(total_len)

df.boxplot(column = 'total_len', by = 'essay_set', figsize = (10, 10))

#get word count discarding punctuations:
def get_word_count(essay):
    return len(re.findall(r"\s", essay))+1

# get sentence
def sent_num(essay):
    sentences_num = len(sent_tokenize(essay))
    return sentences_num

#get sentence length:
def sent_len(essay):
    sentence = sent_tokenize(essay)
    sentence_len = len(word_tokenize(sentence))
    return sentence_len

def get_total_length(essay):
    return len(essay)

def extract_features(essays, feature_functions):
    return [[f(es) for f in feature_functions] for es in essays]


def main():
    print("Reading Training Data")
    training = read_training_data("../Data/training_set_rel3.tsv")
    print("Reading Validation Data")
    test = read_test_data("../Data/valid_set.tsv")

    feature_functions = [get_character_count, get_word_count]

    essay_sets = sorted(training.keys())
    predictions = {}

    for es_set in essay_sets:
        print("Making Predictions for Essay Set %s" % es_set)
        features = extract_features(training[es_set]["essay"],
                                    feature_functions)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(features, training[es_set]["score"])
        features = extract_features(test[es_set]["essay"], feature_functions)
        predicted_scores = rf.predict(features)
        for pred_id, pred_score in zip(test[es_set]["prediction_id"],
                                       predicted_scores):
            predictions[pred_id] = round(pred_score)

    output_file = "../Submissions/length_benchmark.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("prediction_id,predicted_score\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d\n" % (key, predictions[key]))




import nltk
from bs4 import BeautifulSoup
import requests
page = requests.get("http://www.newyorksocialdiary.com/party-pictures/2015/celebrating-the-neighborhood")
soup = BeautifulSoup(page.text, "lxml")
captions = soup.find_all('div', attrs={'class':'photocaption'})
captions.append(soup.find_all('td', attrs={'class':'photocaption'}))

def get_captions(path):
    page = requests.get(path)
    soup = BeautifulSoup(page.text,'lxml')
    names = [c.get_text() for c in soup.find_all('div', attrs={'class':'photocaption'})]
    #captions = soup.find_all('div', attrs={'class':'photocaption'})
    #captions.append(soup.find_all('td', attrs={'class':'photocaption'}))
    #for i in np.arange(0,len(captions)):
        #names.append(captions[i].text)
    #names.append(c.get_text() for c in soup.find_all('td', attrs={'class':'photocaption'}))
    return names

import spacy
captions = get_captions("http://www.newyorksocialdiary.com/party-pictures/2015/celebrating-the-neighborhood")
caption=captions[0]


nlp = spacy.load('en_core_web_sm', disable=['textcat','parser', 'tagger'])
doc = nlp(caption)

names = []
for token in doc.ents:
    if token.label_ =='PERSON' and len(token)>1:
        name = token.text.strip()
        names.append(name)



def extract_entity_names_NE(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names_NE(child))
    return entity_names


import networkx as nx
from collections import Counter








