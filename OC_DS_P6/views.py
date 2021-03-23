from flask import Flask, request
import pandas as pd
import datetime
import pickle
from json import dumps
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import hstack
from .utils import *
from nltk.corpus import stopwords

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

REGEX = app.config['REGEX']
EXTRA_SW = app.config['EXTRA_SW']

# Stopwords nltk
std_sw = set(stopwords.words('english'))

with open(app.config['SOURCE_FILE'], 'rb') as file:
    unpickler = pickle.Unpickler(file)
    tfidf = unpickler.load()
    model = unpickler.load()
    label = unpickler.load()

@app.route('/')
def index():
   return "Générateur de tags StackExchange"

@app.route('/tags/')
def tag_question():
    """
    Prediction function of stackexchange tags from a query passed as parameter
    """

    if len(request.args) != 2:
        result = "Veuillez vérifier les paramètres passés"
    else:
        titre = request.args.get('title')
        corps = request.args.get('body')

        tokenizer = RegexpTokenizer(REGEX)
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        title = clean_field(titre, tknzr=tokenizer, sw=std_sw, \
                            lmtzr=lemmatizer, stmr=stemmer)
        title = ' '.join([w for w in title.split() \
                           if w not in EXTRA_SW and not w.isdigit()])

        body = clean_field(corps, tknzr=tokenizer, sw=std_sw, \
                           lmtzr=lemmatizer, stmr=stemmer)
        body = ' '.join([w for w in body.split() \
                          if w not in EXTRA_SW and not w.isdigit()])

        tfidf_t = tfidf['Title'].transform([title])
        features_t = tfidf['Title'].get_feature_names()

        tfidf_b = tfidf['Body'].transform([body])
        features_b = tfidf['Body'].get_feature_names()

        tfidf_full = hstack([tfidf_t, tfidf_b])

        result = get_tags(label.classes_, model.predict(tfidf_full)[0])

    return dumps({'_Tags': {result}})
