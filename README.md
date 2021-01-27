# Disaster Response Pipeline Project

In this project I apply data engineering skills to create an API that classifies disaster messages from various sources into 36 categories

## Installation

You need python 3 to run this code.

## Project Structure
There are three parts of the project:
### ETL Pipeline
Extract, transform and load the data. This is concerned with processing the data. Namely I loaded, merged and cleaned the messages and categories dataset. I stored into an SQLite database so that the model can use it in the next step to train.
### ML Pipeline
The machine learning pipeline is concerned with training the model and testing it. The pipeline includes a text processing part because it deals with text sources as mentioned in the beginning. I also used GridSearchCV to tune the model further and save it as a pickle file.
### Flask Web App
The run.py process_data and train_classifier are basically the ETL pipeline and ML pipeline included in the terminal work space to make the app work.
