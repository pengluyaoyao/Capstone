import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import re, collections
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict
#df = pd.read_excel("/Users/pengluyao/.kaggle/training_set_rel3.xls")

df = pd.read_excel("training_set_rel3.xls")

#names = list(df.columns.values)
#data = df[['essay_set','essay','domain1_score']].copy()
##print(data)

#df.set_index('essay_id')
#df.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))

#get the essays and essay_set into dictionary
train_data = {}
for index, row in df.iterrows():
    essay = row["essay"].strip()#.split(" ")
    essay_set = row['essay_set']
    domain1_score = row['domain1_score']/2
    if essay_set not in train_data:
        train_data[essay_set] = {"essays":[], "score":[]}

    train_data[essay_set]["essays"].append(essay)
    train_data[essay_set]['score'].append(domain1_score)

#essay1 = train_data[1]['essay'][0]

################################################
##################PREPERATIONS##################
################################################


#exclude = set(string.punctuation)
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


def get_clean_essay(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    return clean_essay

#get list of sentences:
def get_sent_list(essay):
    sent_list = []
    sentences = nltk.sent_tokenize(essay)
    for sentence in sentences:
        clean_sentence = re.sub(r'\W', ' ', str(sentence).lower())
        clean_sentence = re.sub(r'[0-9]', '', clean_sentence)

        sent_list.append(clean_sentence)
    return sent_list

def get_big_dict():
    big = open('big.txt').read()

    words_ = re.findall('[a-z]+', big.lower())

    big_dict = collections.defaultdict(lambda: 0)
    #creating correct word dictionary
    for word in words_:
        big_dict[word] += 1
    return(big_dict)


#######################################
#####FEATURE EXTRACTION FUNCTIONS######
#######################################

#get word count discarding punctuations:
def total_word_count(essay): #excluding punctuations
    list_of_word_list = tokenize(essay)
    flat_list_of_word = [w for l in list_of_word_list for w in l]
    return len(flat_list_of_word)

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


##BOW
from sklearn.feature_extraction.text import HashingVectorizer


def BOW1(essays):
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
    hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, non_negative=True, norm=None, binary=False, ngram_range=(1, 2),stop_words='english')
    hashed_text = hashing_vec.fit_transform(essays)
    hashed_df = pd.DataFrame(hashed_text.data)
    print(hashed_df.head())


def BOW2(essay): ##essay is in the format of df[df['essay_set'] == 1]['essay']
    #sentence = nltk.sent_tokenize(essay)  ##
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=10000,stop_words='english')
    feature_matrix = vectorizer.fit_transform(essay)
    feature_names = vectorizer.get_feature_names()
    return feature_names, feature_matrix

from sklearn.decomposition import NMF

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

loadings=[]
for k in train_data:
    essays = train_data[k]['essays']
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(essays)
    tfidfarray = tfidf.toarray()

    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    loadings.append(np.matmul(tfidfarray, np.transpose(nmf.components_))) #loadings is list of lists



#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)

#######Example of NMF###########################################
#from sklearn.datasets import fetch_20newsgroups
#dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                             remove=('headers', 'footers', 'quotes'))

#data_samples = dataset.data[:n_samples]
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
#                                  max_features=n_features,
#                                  stop_words='english')
#tfidf = tfidf_vectorizer.fit_transform(data_samples)
#nmf = NMF(n_components=n_components, random_state=1,
#          alpha=.1, l1_ratio=.5).fit(tfidf)
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)
###############################################################

##Spelllng errors
def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in big_dict:
            mispell_count += 1

    return mispell_count

###Number of Nouns, Verbs, adj, adv. in an essay
def count_pos(essay):
    tokenized_sentences = tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1

    return noun_count, adj_count, verb_count, adv_count
##adding tags
##sentiment analysis: topic mining. N-gram

def extract_features(essays, feature_functions):
    return [[f(es) for f in feature_functions] for es in essays] ##list of list of features for each essay

feature_functions = [total_word_count, sent_num, word_feature, count_lemmas, count_spell_error, count_pos]

essay_sets = sorted(train_data.keys())

#BOW_dict={}
keys = [1,2,3,4,5,6,7,8]
features = {key: [] for key in keys}
big_dict = get_big_dict()
for es_set in keys:
    #if es_set not in BOW_dict:
     #   BOW_dict={'es_set':[]}
    #BOW_dict['es_set'].append([get_clean_essay(essay) for essay in train_data[es_set]['essays']])

    print("Extracting Features for Essay Set %s" % es_set)
    #if es_set not in features:
        #features={es_set :[]}
    features[es_set].extend(extract_features(train_data[es_set]["essays"], feature_functions))

new_keys = ["total word_count","sentence_number","word_features","count_lemma","spelling_error","count_pos"]
Dict = {key: defaultdict(list) for key in new_keys}
for key, value in features.items():
    for v in value:
        Dict['total word_count']['essay_set %s' % key].append(v[0])
        Dict['sentence_number']['essay_set %s' % key].append(v[1])
        Dict['word_features']['essay_set %s' % key].append(v[2])
        Dict['count_lemma']['essay_set %s' % key].append(v[3])
        Dict['spelling_error']['essay_set %s' % key].append(v[4])
        Dict['count_pos']['essay_set %s' % key].append(v[5])

#######################################################
import dill

Dict = dill.load(open('featurs_dict.pkd', 'rb'))
features = dill.load(open('features.pkd', 'rb'))
Dict['total word_count'].pop('0',None)

#######################################################
#######CREATING DF OF FEATURES AND LABELS##############
#######################################################
dict_todf = {}

for Key in Dict:
    dict_todf[Key]=[]
    for key in Dict[Key]:
        if Dict[Key][key]!=[]:
            dict_todf[Key].extend(Dict[Key][key])



feature_df = pd.DataFrame.from_dict(dict_todf)

###open tuple and list in feature_df
count_tag = {'noun_count':[], 'adj_count':[], 'verb_count':[], 'adv_count':[]}
for t in feature_df['count_pos'].iteritems():
    count_tag['noun_count'].append(t[1][0])
    count_tag['adj_count'].append(t[1][1])
    count_tag['verb_count'].append(t[1][2])
    count_tag['adv_count'].append(t[1][3])

count_tag_df = pd.DataFrame.from_dict(count_tag)

word_features_ = {'avg_word_len':[], 'avg_wordcount':[], 'std_word_count':[]}
for l in feature_df['word_features'].iteritems():
    word_features_['avg_word_len'].append(l[1][0])
    word_features_['avg_wordcount'].append(l[1][1])
    word_features_['std_word_count'].append(l[1][2])

word_features_df = pd.DataFrame.from_dict(word_features_)

from collections import OrderedDict


components = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']

comp_dict = OrderedDict([(c,[]) for c in components])

L = np.arange(0,8)
col=0
for c in comp_dict:
    for l in L:
        comp_by_set = loadings[l][:, col].tolist()
        comp_dict[c].extend(comp_by_set)
    col +=1

BOW_loading_df = pd.DataFrame.from_dict(comp_dict)

full_features_df = pd.concat([feature_df, count_tag_df, word_features_df, BOW_loading_df], axis=1)



############################################
##########MULTICLASS CLASSIFICATIONS########
############################################
'''
n_components can be tuned
'''


############################################
######Bokeh for word count:#################
############################################

from bkcharts import BoxPlot, output_file, show
from bokeh.layouts import row
from numpy import linspace
from scipy.stats.kde import gaussian_kde
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Select
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly
from bokeh.io import curdoc
import colorcet as cc
from bokeh.layouts import widgetbox, row



word_count_dict = Dict['total word_count']
sentence_number_dict = Dict['sentence_number']
word_features_dict = Dict['word_features']
count_lemma_dict = Dict['count_lemma']
spelling_error_dict = Dict['spelling_error']
count_pos_dict = Dict['count_pos']


###############################################




def main():
    print("Reading Training Data")
    training = read_training_data("../Data/training_set_rel3.tsv")
    print("Reading Validation Data")
    test = read_test_data("../Data/valid_set.tsv")

    feature_functions = [total_word_count, sent_num, word_feature, count_lemmas, BOW, count_spell_error, count_pos]

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




