import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function that loads two files with .CSV format , convert them to pandas dataframe,
    merges them together and returns a dataframe
    
    INPUT: message_filepath(str) : String representing the path of the file containing messages
           categories_filepath(str): String representing the path of the file containing categories
           
    OUTPUT: df (dataframe): Dataframe object which is formed after merging two dataframes
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner',left_on='id',right_on='id')
    return df

def clean_data(df):
    """
        Function that takes a pandas dataframe as an input; cleans the dataframe and returns it
        
        INPUT: df (pandas dataframe object) : Dataframe object to be cleaned
        OUTPUT: df (pandas datafrmae object): Cleaned dataframe object
    
    """
    
    # Create a dataframe of 36 individual categories
    categories = df['categories'].str.split(pat=";",expand=True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist()
    categories.columns = category_colnames
    # Loop through each column in categories dataframe
    for column in categories:
        categories[column] = categories[column].astype(str).str.replace(r"[a-z]*-|[a-z]*_","",regex=True)
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df,categories],axis=1,join='inner')
    # Replace 2s with 1 in related column
    df.related.replace(2,1,inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
        Function that takes in a dataframe object and a databse filename and saves the dataframe object into a sql table with the given database filename.
        
        INPUT: df(pandas dataframe object): Dataframe to be saved as a SQL table
               database_filename (str) : String representing the database filename
        OUTPUT: None, Saves the dataframe object into a database as a table.
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_messages_table',if_exists='replace', engine, index=False) 


def main():
    """
        Main Function
    """
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