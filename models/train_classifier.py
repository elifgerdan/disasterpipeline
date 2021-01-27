import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score

def load_data(database_filepath):
    """loads data from sqlite db and splits it into input&output dataframes

    Parameters: 
    database_filepath: sqllite database filepath
    Returns: 
    X: input dataframe
    Y: output dataframe
    cats : category column list
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_cats', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    cats = df.columns[4:]
    return X, Y, cats


def tokenize(text):
    """tokenization function to process text data

    Parameters: 
    text: text to tokenize
    Returns: 
    clean_tokens: tokenized word list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """builds a machine learning pipeline"""
    
    #pipeline = Pipeline([
    #                ('vect', CountVectorizer(tokenizer=tokenize)),
    #                ('tfidf', TfidfTransformer()),
    #                ('clf', MultiOutputClassifier(RandomForestClassifier()))])
        
    #parameters = [{'vect__ngram_range': ((1, 1),(1, 2)),
    #               'vect__max_features': (None, 5000),
    #               'clf__estimator__n_estimators': [10, 100, 250],
    #               'clf__estimator__max_depth':[8],
    #               'clf__estimator__random_state':[42]}]
                  
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = { 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluates the machine learning pipeline
    
    Parameters: 
    model: machine learning model
    X_test: test input dataframe
    Y_test: test output dataframe
    category_names: category name column list
    """
    y_pred = model.predict(X_test)

    def report2dict(cr):
        """converts classification report to dictionary
        
        Parameters: 
        cr: classification report
        Returns:
        class_data: report dictionary
        """
        tmp = list()
        class_data = defaultdict(dict)
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)
        measures = tmp[0]
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return class_data

    for i in range(0,36):
        print('\n' + category_names[i].upper() )
        rpt = report2dict(classification_report(Y_test[:,i], y_pred[:,i] ))
        print ('Precision : {}    \t Recall    : {}    \t F-score   : {}'.format(rpt['avg / total']['precision'], rpt['avg / total']['recall'], rpt['avg / total']['f1-score']))



def save_model(model, model_filepath):
    """saves the machine learning model to pickle file
    
    Parameters: 
    model: machine learning model
    model_filepath: pickle file path
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print(datetime.now())
        model.fit(X_train, Y_train)
        print(datetime.now())
        print('best score: ', model.best_score_, 'best params: ', model.best_params_)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()