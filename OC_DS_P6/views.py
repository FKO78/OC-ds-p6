import pandas as pd
import datetime
from flask import Flask, request #, render_template, url_for,
from .utils import *
from pickle import Unpickler
from json import dumps
#import sys

#sys.setrecursionlimit(20000)

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']


with open(app.config['SOURCE_FILE'], 'rb') as file:
    unpickler = pickle.Unpickler(file)
    sclr = unpickler.load()
    tfidf = unpickler.load()
    mod = unpickler.load()
    lbl = unpickler.load()

@app.route('/')
def index():
   return "Générateur de tags StackExchange"

@app.route('/tags/')
def tag_question(title, body):
    """
    Prediction function of stackexchange tags from a query passed as parameter
    """

    RATIO = 1/3
    REGEX = '[a-z0-9]+[#-]?[a-z0-9]*'

    # Stopwords nltk
    std_sw = set(nltk.corpus.stopwords.words('english'))

    # Extra stopwords = radicaux qui ne me semblent pas discriminants
    extra_sw = ('use', 'get', 'like', 'way', 'creat', 'would', 'want', 'need',\
                'know', 'could', 'x', 'xx', 'xyz', 'aa', 'xxx', 'z', 'yyyi', 'wont',\
                'aaa', 'aaaaaa', 'aabbc', 'aandb', 'aarrggbb')


    tokenizer = nltk.RegexpTokenizer(REGEX)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    with open('OC_DS_P6_prod.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        sclr = unpickler.load()
        tfidf = unpickler.load()
        mod = unpickler.load()
        lbl = unpickler.load()

    title = clean_field(title, tknzr=tokenizer, sw=std_sw, \
                        lmtzr=lemmatizer, stmr=stemmer)
    title = ' '.join([w for w in title.split() \
                       if w not in extra_sw and not w.isdigit()])

    body = clean_field(body, tknzr=tokenizer, sw=std_sw, \
                       lmtzr=lemmatizer, stmr=stemmer)
    body = ' '.join([w for w in body.split() \
                      if w not in extra_sw and not w.isdigit()])

    tfidf_q = tfidf['Title'].transform([title])
    cols = tfidf['Title'].get_feature_names()
    tfidf_q = pd.DataFrame(tfidf_q.todense().tolist(), columns=cols)

    tfidf_b = tfidf['Body'].transform([body])
    cols = tfidf['Body'].get_feature_names()
    tfidf_b = pd.DataFrame(tfidf_b.todense().tolist(), columns=cols)

    X = (RATIO * tfidf_q).add((1 - RATIO) * tfidf_b, fill_value=0).fillna(0)

    return '<{}>'.format('><'.\
                         join(lbl.inverse_transform(mod.predict(sclr.transform(X)))))
