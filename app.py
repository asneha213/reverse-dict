#----------------------------------------------------------------------
# Author: Sneha Reddy Aenugu
# Description: Web application for reverse dictionary
#-----------------------------------------------------------------------

import os
import argparse
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import pandas as pd
from pathlib import Path
import pickle

import faiss
from nltk.corpus import wordnet as wn
import torch
from src.generate_word_prompts import *
from src.create_embeddings import *


df = pd.read_pickle("store/df.pkl")


def create_app(model, num_entries, config=None):
    app = Flask(__name__)

    # See http://flask.pocoo.org/docs/latest/config/
    app.config.update(dict(DEBUG=True))
    app.config.update(config or {})

    # Setup cors headers to allow all domains
    # https://flask-cors.readthedocs.io/en/latest/
    CORS(app)

    # Definition of the routes. Put them into their own file. See also
    # Flask Blueprints: http://flask.pocoo.org/docs/latest/blueprints

    @app.route('/')
    def home():
        return render_template('home.html')


    @app.route('/predict',methods=['POST'])
    def predict():
        if request.method == 'POST':
            message = request.form['message']
            list_defns = get_word_prompts_from_query(df, message, model, model_code, num_entries)
        return render_template('result.html',prediction = list_defns)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default="8000")
    parser.add_argument('--model_code', dest='model_code', default='dsbert')
    parser.add_argument('--num_entries', dest='num_entries', default=20)

    args = parser.parse_args()
    port = int(args.port)

    model_code = args.model_code
    num_entries = int(args.num_entries)

    model = load_model(model_code)
    app = create_app(model, num_entries)
    app.run(host="0.0.0.0", port=port)
