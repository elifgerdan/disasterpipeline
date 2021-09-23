# Table of Contents

1. [Project Motivation](#disaster-response-pipeline-project-motivation)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Stracture & File Descriptions](#project-structure)
4. [Results](#results)
6. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

# Disaster Response Pipeline Project Motivation

In this project I apply data engineering skills to create an API that classifies disaster messages from various sources into 36 categories. You can see the categoical distribution of the training data below;
![alt text](https://github.com/elifgerdan/disasterpipeline/blob/main/cats.png?raw=true)

## Installation
Main libraries you may need for this project to run flawlessly are; numpy, pandas, json, sklearn, sqlite3, sqlalchemy, plotly, flask, collections, pickle, re. you can see a full list of them below. You may use the package manager [pip](https://pip.pypa.io/en/stable/) to install if any of them is missing.

## Usage
```python
import json
import nltk
import numpy as np
import numpy as np 
import pandas as pd
import pickle
import plotly
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from flask import Flask
from flask import render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Pie, Heatmap, Margin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine
```

## Project Structure
There are three parts of the project:
### ETL Pipeline
- [process_data.py](https://github.com/elifgerdan/disasterpipeline/blob/main/data/process_data.py)  -  Extract, transform and load the data. This is concerned with processing the data. Namely I loaded, merged and cleaned the messages and categories dataset. I stored into an SQLite database so that the model can use it in the next step to train. 
![alt text](https://github.com/elifgerdan/disasterpipeline/blob/main/heat.png?raw=true)

### ML Pipeline
- [train_classifier.py](https://github.com/elifgerdan/disasterpipeline/blob/main/models/train_classifier.py)  -  The machine learning pipeline is concerned with training the model and testing it. The pipeline includes a text processing part because it deals with text sources as mentioned in the beginning. I also used GridSearchCV to tune the model further and save it as a pickle file.
### Flask Web App
- [run.py](https://github.com/elifgerdan/disasterpipeline/blob/main/app/run.py)  -  The run.py process_data and train_classifier are basically the ETL pipeline and ML pipeline included in the terminal work space to make the app work.
## Results
This classification problem is a type of supervised machine learning because the model learns to classify the outcome based on learning from the data given to it. I have managed to extract and clean the data and load it into an sqlite db. I then process the messages using nlp techniques, after that I used multi level classification machine learning technique to predict on the cleaned-tokenized training dataset. Finally I put the results on the flask app to explore. :)

## Licensing, Authors, Acknowledgements
Thank you to FigureEight (now appen) for providing this meaningful dataset and Udacity for Data Science Nanodegree program based on real problem and use cases.
