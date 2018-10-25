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
df_test = pd.read_excel("valid_set.xls")
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

test_data = {}
for index, row in df_test.iterrows():
    essay = row["essay"].strip()#.split(" ")
    essay_set = row['essay_set']
    if essay_set not in test_data:
        test_data[essay_set] = {"essays":[]}

    test_data[essay_set]["essays"].append(essay)


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

    lemmas = set()
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.add(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.add(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.add(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.add(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.add(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(lemmas)

    return lemma_count

##BOW
from sklearn.feature_extraction.text import HashingVectorizer

'''
def BOW1(essays):
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
    hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, non_negative=True, norm=None, binary=False, ngram_range=(1, 2),stop_words='english')
    hashed_text = hashing_vec.fit_transform(essays)
    hashed_df = pd.DataFrame(hashed_text.data)
    print(hashed_df.head())
'''

def BOW2(essay): ##essay is in the format of df[df['essay_set'] == 1]['essay']
    #sentence = nltk.sent_tokenize(essay)  ##
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=1000,stop_words='english')
    feature_matrix = vectorizer.fit_transform(essay)
    feature_names = vectorizer.get_feature_names()
    return feature_names, feature_matrix

'''
from sklearn.decomposition import NMF

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

n_samples = 2000
n_features = 100
n_components = 20
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
'''


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

def extract_features(essays, feature_functions):
    return [[f(es) for f in feature_functions] for es in essays] ##list of list of features for each essay

feature_functions = [total_word_count, sent_num, word_feature, count_lemmas, count_spell_error, count_pos]

def creating_features(data):
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
        features[es_set].extend(extract_features(data[es_set]["essays"], feature_functions))

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
    return features, Dict

feature_train, Dict_train = creating_features(train_data)
feature_test, Dict_test = creating_features(test_data)


#######################################################
import dill

Dict = dill.load(open('featurs_dict.pkd', 'rb'))
features = dill.load(open('features.pkd', 'rb'))
Dict['total word_count'].pop('0',None)

#dill.dump(feature_test, open('features_test.pkd', 'wb'))
#dill.dump(Dict_test, open('featurs_dict_test.pkd', 'wb'))

Dict_test = dill.load(open('featurs_dict_test.pkd', 'rb'))
features_test = dill.load(open('features_test.pkd', 'rb'))


#######################################################
#######CREATING DATAFRAME OF FEATURES AND LABELS#######
#######################################################

def creating_df_from_feature_dict(Dict):
    dict_todf = {}
    for Key in Dict:
        dict_todf[Key]=[]
        for key in Dict[Key]:
            if Dict[Key][key]!=[]:
                dict_todf[Key].extend(Dict[Key][key])
    return dict_todf

dict_todf = creating_df_from_feature_dict(Dict)
dict_todf_test = creating_df_from_feature_dict(Dict_test)

feature_df = pd.DataFrame.from_dict(dict_todf)
feature_df_test = pd.DataFrame.from_dict(dict_todf_test)

###open tuple and list in feature_df
####count of noun etc df, open tuple

def open_tuple(feature_df):
    count_tag = {'noun_count':[], 'adj_count':[], 'verb_count':[], 'adv_count':[]}
    for t in feature_df['count_pos'].iteritems():
        count_tag['noun_count'].append(t[1][0])
        count_tag['adj_count'].append(t[1][1])
        count_tag['verb_count'].append(t[1][2])
        count_tag['adv_count'].append(t[1][3])
    return count_tag

count_tag_test = open_tuple(feature_df_test)
count_tag = open_tuple(feature_df)
count_tag_df = pd.DataFrame.from_dict(count_tag)
count_tag_df_test = pd.DataFrame.from_dict(count_tag_test)

####word features df, open list

def open_list(feature_df):
    word_features_ = {'avg_word_len':[], 'avg_wordcount':[], 'std_word_count':[]}
    for l in feature_df['word_features'].iteritems():
        word_features_['avg_word_len'].append(l[1][0])
        word_features_['avg_wordcount'].append(l[1][1])
        word_features_['std_word_count'].append(l[1][2])
    return word_features_

word_features_ = open_list(feature_df)
word_features_df = pd.DataFrame.from_dict(word_features_)

word_features_test = open_list(feature_df_test)
word_features_df_test= pd.DataFrame.from_dict(word_features_test)

'''
#####NMF loading df
from collections import OrderedDict

components = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20']

comp_dict = OrderedDict([(c,[]) for c in components])

L = np.arange(0,8)
col=0
for c in comp_dict:
    for l in L:
        comp_by_set = loadings[l][:, col].tolist()
        comp_dict[c].extend(comp_by_set)
    col +=1

BOW_loading_df = pd.DataFrame.from_dict(comp_dict)
'''

####Using original tfidfarray instead of loadings on 10 components by NMF
####Cannot convert to df from dict because each essay set has different BOW df (feature names are different).
####concat BOW df essay set by essay set

from collections import OrderedDict

def BOWvectorizer_toDict(essays):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=500,
                                   stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(essays)
    tfidfarray = tfidf.toarray()  ##n_essays in each set *n_features array
    words = tfidf_vectorizer.get_feature_names()
    word_dict = OrderedDict([(w,[]) for w in words])
    col=0
    for w in word_dict:
        word_freq = tfidfarray[:,col].tolist()
        word_dict[w].extend(word_freq)
    col+=1
    return word_dict

def create_wc_dict(dict):
    full_word_count_dict = {}
    Sets=np.arange(1,9)
    for Set in Sets:
        essays = dict[Set]['essays']
        word_count_dict = BOWvectorizer_toDict(essays)
        full_word_count_dict[Set]=word_count_dict
    return full_word_count_dict

full_word_dict_test = create_wc_dict(test_data)

#dill.dump(full_word_dict, open('full_word_dict.pkd', 'wb'))
full_word_dict = dill.load(open('full_word_dict.pkd', 'rb'))

#dill.dump(full_word_dict_test, open('full_word_dict_test.pkd', 'wb'))
full_word_dict_test = dill.load(open('full_word_dict_test.pkd', 'rb'))

##########Full features df
full_features_df = pd.concat([df['essay_set'], feature_df, count_tag_df, word_features_df], axis=1)
full_features_df_test = pd.concat([df_test['essay_set'], feature_df_test, count_tag_df_test, word_features_df_test], axis=1)

labels =pd.concat([df['essay_set'], df['domain1_score']], axis=1)

#dill.dump(full_features_df, open('full_features_df.pkd', 'wb'))
#dill.dump(labels, open('labels.pkd', 'wb'))
full_features_df = dill.load(open('full_features_df.pkd', 'rb'))
labels = dill.load(open('labels.pkd', 'rb'))

#dill.dump(full_features_df_test, open('full_features_df_test.pkd', 'wb'))
full_features_df_test= dill.load(open('full_features_df_test.pkd', 'rb'))
#####Train and Test Split###########



###############################################################################
################                                        ########################
################    MODEL FITTING AND CROSSVALIDATION   ########################
################                                        #######################
###############################################################################


############################################################################
##########   CROSS-VALIDATION INITIALIZATION   ################################
############################################################################
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn import ensemble
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
'''
def reg_grid_cv(X, y, clf, params):
    estimator = GridSearchCV(clf, params, cv=3)
    estimator.fit(X, y)
    return estimator.best_score_, estimator.best_estimator_#.feature_importance() #estimator.cv_results_['mean_test_score']
'''

def reg_rnd_cv(X, y, clf, n_iter_search, param_dist, n_jobs):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=3, n_jobs=n_jobs)
    random_search.fit(X, y)
    return random_search.best_score_, random_search.best_estimator_


##########################CROSS-VALIDATION FOR EACH ESSAY SET#################################
####################DEFINING PARAMETERS TO BE TUNED: classifier dictionary INCLUDE 'GBR'####################

classifiers= {#'rf': {'clf': RandomForestRegressor(), 'params': {'n_estimators': np.arange(170, 180, 5)}},
              'gbr':{'clf': ensemble.GradientBoostingRegressor(),
                     '1': {'n_estimators':np.arange(80,400,30), 'max_depth': np.arange(1,20,2),
                         'min_samples_split': [0.3,0.4,0.5,0.6,0.7,0.8],'learning_rate':np.arange(0.05,0.2,0.02), 'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '2': {'n_estimators':np.arange(80,400,30), 'max_depth': np.arange(1,20,2),
                         'min_samples_split': [0.3,0.4,0.5,0.6,0.7,0.8],'learning_rate':np.arange(0.05,0.2,0.02),'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '3': {'n_estimators':np.arange(80,400,30), 'max_depth': np.arange(1,20,2),
                         'min_samples_split': [0.3,0.4,0.5,0.6,0.7,0.8],'learning_rate':np.arange(0.05,0.2,0.02),'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '5': {'n_estimators':np.arange(80,400,30), 'max_depth': np.arange(20,50,5),
                         'min_samples_split': [0.3,0.4,0.5,0.6,0.7,0.8],'learning_rate':np.arange(0.05,0.2,0.02),'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '6': {'n_estimators':np.arange(80,400,30), 'max_depth': np.arange(1,20,2),
                         'min_samples_split': np.arange(0.05, 0.4,0.05),'learning_rate':np.arange(0.05,0.2,0.02),'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '7': {'n_estimators':np.arange(400,900,100), 'max_depth': np.arange(1,20,2),
                         'min_samples_split': np.arange(0.05, 0.4,0.05),'learning_rate':np.arange(0.001,0.011,0.002),'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]},
                     '8': {'n_estimators':np.arange(400,900,100), 'max_depth': np.arange(12,20,2),
                         'min_samples_split': np.arange(0.6,1,0.1),'learning_rate':np.arange(0.05,0.2,0.02), 'validation_fraction': [0.2], 'n_iter_no_change': [10], 'tol': [0.01]}
                     }
              }

Sets=[1,2,3,5,6,7,8]



def create_Xy_to_train_model(feature_df, BOW_Dict, score_df, Sets):
    X_all_sets = {}
    y_all_sets = {}
    for Set in Sets:
        BOW_freq_df = pd.DataFrame.from_dict(BOW_Dict[Set])
        index = BOW_freq_df.index.values
        features_df_bySet = feature_df[(feature_df['essay_set'] == Set)].set_index(index)
        full_features_df_bySet = pd.concat([features_df_bySet, BOW_freq_df], axis=1)
        X_train = full_features_df_bySet.drop(['essay_set', 'count_pos', 'word_features'], axis=1)
        X_train = X_train.as_matrix()  ##X_train is an np array
        y_train = score_df[(score_df['essay_set'] == Set)]['domain1_score'].tolist()##y_train is a list
        X_all_sets[Set] = X_train
        y_all_sets[Set] = y_train
    return X_all_sets, y_all_sets

X_train, y_train = create_Xy_to_train_model(full_features_df, full_word_dict, labels, [1,2,3,5,6,7,8])


clfs = ['gbr']
best_scores={Set:{clf:[] for clf in clfs} for Set in Sets}
for clf in clfs:
    for Set in Sets:
    #best_score = reg_grid_cv(X_train, y_train, classifiers[clf]['clf'], classifiers[clf]['params'])
        best_score = reg_rnd_cv(X=X_train[Set], y=y_train[Set], clf=classifiers['gbr']['clf'], n_iter_search=20, param_dist= classifiers['gbr'][str(Set)], n_jobs=3)
        print(best_score)
        best_scores[Set][clf].append(best_score)


#########################XGBOOST###########
'''
clfs = ['gbr']
best_scores={Set:{clf:[] for clf in clfs} for Set in Sets}

def train_xgb(feature_df, BOW_Dict, score_df, Sets):
    for Set in Sets:
        BOW_freq_df = pd.DataFrame.from_dict(BOW_Dict[Set])
        index = BOW_freq_df.index.values
        features_df_bySet = feature_df[(feature_df['essay_set'] == Set)].set_index(index)
        full_features_df_bySet = pd.concat([features_df_bySet, BOW_freq_df], axis=1)
        X= full_features_df_bySet.drop(['essay_set', 'count_pos', 'word_features'], axis=1)
        y = score_df[(score_df['essay_set'] == Set)]['domain1_score'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(y_train)
        #evallist = [(dtest, 'eval'), (dtrain, 'train')]
        xgb_reg = xgb.XGBRegressor(max_depth=3, n_estimators=700, learning_rate=0.1)
        xgb_reg.fit(X_train, y_train, eval_metric='rmse', early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)#eval_set=[(X_train, y_train),(X_test, y_test)], verbose=True
        #predictions = xgb_reg.predict(X_test)
        pred = xgb_reg.predict(X_test, y_test, ntree_limit=xgb_reg.best_ntree_limit)
    return    explained_variance_score(pred,y_test)



train_xgb(full_features_df, full_word_dict, labels, [2])
'''



###############regression model comparisons bar plot###############################
import matplotlib.pyplot as plt

bar_offsets = (np.arange(len(Sets))*(len(clfs) + 1) + .5)
plt.figure()
COLORS = 'bgrcmyk'

for i, clf in enumerate(clfs):
    clf_best_scores=[]
    for set in sets:
        clf_best_scores.extend(best_scores[set][clf])
        clf_best_scores_array = np.asarray(clf_best_scores)
    plt.bar(bar_offsets + i/2, clf_best_scores_array, label=clf, color=COLORS[i])

plt.title("Comparing Different Regression Predictions")
plt.xlabel('Essay Sets')
plt.xticks(bar_offsets + 3/2, sets)
plt.ylabel('R square')
plt.ylim((0, 1))
plt.legend(loc='upper right')

plt.show()


############feature importance plot: gbr in essay set 1 with best estimator#############
'''
essay1 and essay 5 have better predictions, lets take a look at the feature importance
'''
sets=np.array([1,5])

for set in sets:
    #set = 1
    BOW_freq_df = pd.DataFrame.from_dict(full_word_dict[set])
    index = BOW_freq_df.index.values
    features_df_bySet = full_features_df[(full_features_df['essay_set'] == set)].set_index(index)
    full_features_df_bySet = pd.concat([features_df_bySet, BOW_freq_df], axis=1)
    X_train = full_features_df_bySet.drop(['essay_set', 'count_pos', 'word_features'], axis=1)
    X_train = X_train.as_matrix()  ##X_train is an np array
    y_train = labels[(labels['essay_set'] == set)]['domain1_score'].tolist()  ##y_train is a list
    gbr = ensemble.GradientBoostingRegressor()
    grid = GridSearchCV(gbr, {'n_estimators': [50, 100, 500, 1000], 'learning_rate': [1, 0.1, 0.3, 0.01],
                              'loss': ['ls']})
    grid.fit(X_train, y_train)
    importance = grid.best_estimator_.feature_importances_
    rel_importance = 100.0 * (importance / importance.max())
    feature_names = np.asarray(full_features_df_bySet.iloc[:, 1:].columns.values)
    sorted_idx = np.argsort(rel_importance)
    sorted_idx = sorted_idx[91:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, rel_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

############################################################################################################
###########CHECKING THE OVERFITTING OF GRADIENT BOOSTING REGRESSION##############################
##################################################################################################
from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve

def GradientBooster(param_grid):
    estimator = ensemble.GradientBoostingRegressor()
    #cv = ShuffleSplit(X_train.shape[0], test_size=0.3)
    classifier = GridSearchCV(estimator=estimator, cv=3, param_grid=param_grid)
    classifier.fit(X_train, y_train)
    print ("Best Estimator learned through GridSearch")
    print(classifier.best_estimator_, classifier.best_score_)
    return classifier.best_estimator_, classifier.best_score_

param_grid = {'n_estimators':[50, 100, 500, 750, 1000], 'learning_rate':[1, 0.1, 0.3, 0.01, 0.005], 'max_depth':[6, 4]}



for set in sets:
    #set = 8
    BOW_freq_df = pd.DataFrame.from_dict(full_word_dict[set])
    index = BOW_freq_df.index.values
    features_df_bySet = full_features_df[(full_features_df['essay_set'] == set)].set_index(index)
    full_features_df_bySet = pd.concat([features_df_bySet, BOW_freq_df], axis=1)
    X_train = full_features_df_bySet.drop(['essay_set', 'count_pos', 'word_features'], axis=1).as_matrix() ##X is an np array
    y_train = labels[(labels['essay_set'] == set)]['domain1_score'].tolist()  ##y is a list
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    best_est, best_score = GradientBooster(param_grid)

title = "Learning Curves (Gradient Boosted Regression Trees)"
estimator1 = ensemble.GradientBoostingRegressor(n_estimators=750, max_depth=4,  learning_rate=0.005)
estimator2 = ensemble.GradientBoostingRegressor(n_estimators=750, max_depth=4,  learning_rate=0.005)
estimator3 = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4,  learning_rate=0.002)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

plot_learning_curve(estimator2, title, X_train, y_train, cv=5, n_jobs=1)
plt.title('Learning Curves (Essay_set 8): n_estimators=750, max_depth=4,  learning_rate=0.005')
plt.show()



import scholarly
search_query = scholarly.search_pubs_query('Chaos theory and strategy: Theory, application, and managerial implications')
author = next(search_query).fill()
results = [print(citation.bib['title']) for citation in author.get_citedby()]

results = [citation.bib['title'] for citation in author.get_citedby()]


