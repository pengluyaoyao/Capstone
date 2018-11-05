
import numpy as np
import dill
from flask import Flask, render_template, request
import re, collections
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy import stats
import os


bokeh_app= Flask(__name__)

##################################################
###############FIGURE CREATION ###################
##################################################
'''
def styling(figure):
    figure.title.text = "Feature Index Comparison with the Population"
    figure.title.align = 'center'
    figure.title.text_font_size = '20pt'
    figure.title.text_font = 'serif'

    # Axis titles
    figure.xaxis.axis_label_text_font_size = '14pt'
    figure.xaxis.axis_label_text_font_style = 'bold'
    figure.yaxis.axis_label_text_font_size = '14pt'
    figure.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    figure.xaxis.major_label_text_font_size = '12pt'
    figure.yaxis.major_label_text_font_size = '12pt'

    return figure


def joy(category, data, scale):
    y = [(category, 0)]
    y.extend(list(zip([category]*len(data), scale*data)))
    return y


def joy(category, data, scale):
    f = data
    f[0]=0
    return list(zip([category]*len(data), scale*f))

def create_figures(data, features):
    p1 = figure(y_range=features, plot_height=500, plot_width=800, x_range=(0, 600), toolbar_location='below')
    xMax = []
    for i, feature in enumerate(features):
        pdf = gaussian_kde(data[feature], 'silverman')
        #feature_dict = data_dict[cat]
        xmax = max(data[feature])
        x = (linspace(0, xmax, 500))
        source = ColumnDataSource(data={'x': x})
        y = joy(feature, pdf(x), xmax / 5)
        source.add(y, feature)
        palette = [cc.rainbow[i * 15] for i in range(17)]
        p1.patch('x', feature, fill_color=palette[i], fill_alpha=0.8, line_color=None, source=source)
        xMax.append(xmax)
    p1.x_range = Range1d(0,max(xMax))
    return styling(p1)
'''

def data_prep(data, category):
    col_names = ['essay_set', 'count_lemma', 'sentence_number', 'spelling_error', 'total word_count', 'adj_count',
                'adv_count', 'noun_count',
                'verb_count', 'avg_word_len', 'avg_wordcount', 'std_word_count']
    #features_word = ['count_lemma', 'total word_count', 'adj_count', 'adv_count', 'noun_count',
    #                 'verb_count', 'avg_word_len', 'avg_wordcount', 'std_word_count']
    #features_sentence = ['sentence_number', 'spelling_error']
    if category == 'Prompt1':
        cat = 1
    else :
        cat = 5
    feature_1or5_df = data.loc[data['essay_set'] == cat][col_names]
    #dict_1_5 = {f : {'essay_set1' : [], 'essay_set5': []} for f in colnames[1:]}

    #for K, V in dict_1_5.items():
    #   dict_1_5[K]['essay_set1'] = feature_1_5_df.loc[feature_1_5_df['essay_set'] ==1][K]
    #    dict_1_5[K]['essay_set5'] = feature_1_5_df.loc[feature_1_5_df['essay_set'] ==5][K]
    return feature_1or5_df


################################################
######ALGORITHMS OF FEATURE EXTRACTIONS##########
#################################################


'''
PREPERATIONS
'''
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

def get_big_dict():
    big = open('big.txt').read()

    words_ = re.findall('[a-z]+', big.lower())

    big_dict = collections.defaultdict(lambda: 0)
    #creating correct word dictionary
    for word in words_:
        big_dict[word] += 1
    return big_dict

'''
FEATURE EXTRACTIONS FUNCTIONS
'''

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

def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()
    big_dict = get_big_dict()
    for word in words:
        if not word in big_dict:
            mispell_count += 1

    return mispell_count

'''
def spell_errors(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispells = []

    words = clean_essay.split()
    big_dict = get_big_dict()
    for word in words:
        if not word in big_dict:
            mispells.append(word)
    return {'spelling_errors' : mispells}
'''

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

    return [noun_count, adj_count, verb_count, adv_count]

'''
FEATURE EXTRACTION
'''
feature_functions = [total_word_count, sent_num, word_feature, count_lemmas, count_spell_error, count_pos]

def create_features(essay, feature_functions):
    features_no_clean = [f(essay) for f in feature_functions]
    feature_names = ['total word count', 'sentence number', 'average word length', 'average wordcount', 'SD of wordcount', 'count of lemmas', 'number of spelling errors', 'number of nouns','number of adjectives','number of verbs','number of adverbs']
    feature_values_list = []
    for F in features_no_clean:
        if type(F) != list:
            feature_values_list.append(round(F,2))
        else:
            for f in F:
             feature_values_list.append(round(f,2))
    feature_values_dict = dict(zip(feature_names, feature_values_list))
    return feature_values_list, feature_values_dict

'''
PREDICTIONS USING GRADIENT BOOSTING REGRESSOR
'''
import pickle

def predictions_gbr(x_test, category):
    labels = dill.load(open('labels.pkd', 'rb'))
    if category[0] == 'Prompt1':
        score_model = pickle.load(open('score_model_set1.pickle', 'rb'))
        pred = score_model.predict(x_test)
        y_upper = pred+2*(0.6106829419850511**0.5)
        y_lower = pred-2*(0.6106829419850511**0.5)
        y_train = labels.loc[labels['essay_set']==1]['domain1_score']
    else:
        score_model = pickle.load(open('score_model_set5.pickle', 'rb'))
        pred = score_model.predict(x_test)
        y_upper = pred+2*(0.28391495746239453**0.5)
        y_lower = pred-2*(0.28391495746239453**0.5)
        y_train = labels.loc[labels['essay_set'] == 5]['domain1_score']
    return {'predicted score: ': round(pred[0]), 'grade interval (90%)': (round(y_lower[0]), round(y_upper[0])), 'percentile': round(stats.percentileofscore(y_train, pred),2)}

##################################################################################################
#################################################################################################
###############################  APP DEVELOPMENT (PUTTING TOGETHER) ###############################
##################################################################################################
##################################################################################################

@bokeh_app.route('/')
def index():
    return render_template('index.html')


@bokeh_app.route('/results', methods=['get','post'])

def show_results():
    full_feature = dill.load(open('full_features_df.pkd', 'rb'))
    category = request.form.getlist('check')
    #feature_1or5_df = data_prep(full_feature, category)
    #features  = request.form.getlist('select')
    new_essay = request.form['text']
    feature_values_list, feature_values_dict = create_features(new_essay, feature_functions)

    #plot = create_figures(feature_1or5_df, features)

    predictions_dict = predictions_gbr(np.reshape(feature_values_list, (1,-1)), category)

    #script2, div2 = components(plot)
    text = '"%s"' % new_essay
    s = os.popen("echo %s | pylanguagetool" % text).read()
    clipboard = s.replace('\n', '<br />')

    return render_template('about.html', clipboard = "%s" % clipboard, predictions_dict = predictions_dict, feature_values_dict = feature_values_dict)#, script2=script2, div2=div2) #ticker_name=ticker_name, col_active0=col_active0) #col_active1=col_active1,



if __name__ == '__main__':
    bokeh_app.run(port=5000)


'''

layout = column(widgetbox(select1), p)

curdoc().add_root(layout)
'''