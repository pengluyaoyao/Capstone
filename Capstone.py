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
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#df = pd.read_excel("/Users/pengluyao/.kaggle/training_set_rel3.xls")

df = pd.read_excel("training_set_rel3.xls")

names = list(df.columns.values)
data = df[['essay_set','essay','domain1_score']].copy()
#print(data)

df.set_index('essay_id')
df.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))

#get the essays and essay_set into dictionary
train_data = {}
for index, row in df.iterrows():
    essay = row["essay"].strip()#.split(" ")
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

#essay1 = train_data[1]['essay'][0]

###PREPERATIONS
exclude = set(string.punctuation)
  #each period is a sentence


#def excluding_punctuation(essay):
 #   essay_no_punct = ''.join(ch for ch in essay if ch not in exclude)
  #  return essay_no_punct

def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)

    return tokens


def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    return tokenized_sentences

#get list of sentences:
def get_sent_list(essay):
    sent_list = []
    sentences = nltk.sent_tokenize(essay)
    for sentence in sentences:
        sent_list.append(''.join([ch for ch in sentence if ch not in exclude]))
    return sent_list

#####FEATURE EXTRACTION FUNCTIONS
#get word count discarding punctuations:
def total_word_count(essay): #excluding punctuations
    list_of_word_list = tokenize(essay)
    return len(list_of_word_list)

# get sentence
def sent_num(essay):  #number of sentences in an essay
    sentences_num = len(sent_tokenize(essay))
    return sentences_num

def word_feature(essay): #average word count and std of word count in sentence, avg word length throughout an essay
    word_len =[]
    words_in_sent = tokenize(essay)
    for sent in words_in_sent:
        word_len.extend([len(word) for word in sent])
    avg_word_len = np.mean(word_len)
    word_count_per_sentence = [len(s) for s in words_in_sent]
    avg_wordcount = np.mean(word_count_per_sentence)
    std_word_count =  np.std(word_count_per_sentence) #by sentence
    return [avg_word_len, avg_wordcount, std_word_count]


##number of lemmas:
def count_lemmas(essay):
    tokenized_sentences = tokenize(essay)

    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count

##Spellilng errors



##adding tags
##sentiment analysis: topic mining. N-gram


#

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




