import os
from flask import Flask
import pandas as pd
import datetime
from flask import Flask, request #, render_template, url_for,
from pickle import Unpickler
from json import dumps
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import hstack
from .views import app
from .utils import *

#from . import models

# Connect sqlalchemy to app
#models.db.init_app(app)

#@app.cli.command()
#def init_db():
#    models.init_db()
