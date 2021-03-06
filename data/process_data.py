import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import re

def load_data(messages_filepath, categories_filepath):
    """loads both datasets from csv and merge into one df

    Parameters: 
    messages_filepath (str): location path of message csv
    categories_filepath (str): location path of category csv 
    Returns: 
    df: dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',left_on=['id'],right_on=['id'])

    return df


def clean_data(df):
    """cleans loaded df, splits into columns arranges column names 
       and filters out unneccessary rows

    Parameters: 
    df: dataframe that cleaning process to be applied to
    Returns: 
    df: cleaned dataframe
    """
    
    categories = pd.DataFrame(df.categories.str.split(';',36).tolist())
    row = categories.iloc[0]
    category_colnames = pd.Series(row.values).apply(lambda d: d.split('-')[0]).values
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = pd.to_numeric(categories[column])

    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df['message'] = df['message'].apply(lambda x: x.lower())
    
    for c in category_colnames:
        df = df[df[c] != 2]
        
    return df

def save_data(df, database_filename):
    """saves data into SQLlite DB

    Parameters: 
    df: dataframe that has the relevant data
    database_filename: sqllite db path to put data in
    """
    engine = create_engine('sqlite:///' + database_filename)

    conn = sqlite3.connect(database_filename)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS message_cats")
    conn.commit()
    conn.close()

    df.to_sql('message_cats', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()