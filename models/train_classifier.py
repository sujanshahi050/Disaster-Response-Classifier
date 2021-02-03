import re
import sys
import pickle
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])




def load_data(database_filepath):
    """
        Function to load data from a sql database and convert them into dataframe
        
        INPUT: database_filepath(str): String representation of the filepath of a database
        OUTPUT: X(DataFrame Object): Returns a dataframe object of messages
                y (Series) : Returns a Series object containing the categories
    """
    table_name = 'disaster_messages_table'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.drop(["message","id","genre","original"],axis=1)
    category_names = y.columns
    return X,y,category_names


def tokenize(text):
    """
        Function that takes a text(str) as an input, cleans it and returns a lemmatized list of tokens(words)
        
        INPUT: text (str) : Text to be cleaned and tokenized and lemmatized
        OUTPUT: clean_text(list): List of clean tokens extracted from the text.
    """
    
    # Regex Pattern to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    # Replace all urls with 'urlplaceholder'
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    tokens_list = word_tokenize(text)
    # Instance of a Lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_text = []
    for token in tokens_list:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_text.append(clean_token)
    return clean_text

    
def build_model():
    """
        Function to build a model based on a machine learning algorithm to do classification.
        Using AdaBoostClassifer for this project
        INPUT: None
        OUTPUT: cv (ML model) : Machine Learning model to classify a category from a given text
    """
    
    # Pipeline to create a AdaBoost Classifier
    pipeline_ada =  Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Parameters for AdaBoost Classifier
    parameters_ada = {'clf__estimator__n_estimators': [50,100],
                     }
    cv = GridSearchCV(pipeline_ada,param_grid=parameters_ada,cv=2,verbose=3,n_jobs=-1)
    return cv

def plot_f1_scores(y_test,y_pred):
    i=0
    for col in y_test:
        print('Feature{}:{}'.format(i+1,col))
        print(classification_report(y_test[col],y_pred[:,i]))
        i = i + 1
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))



def evaluate_model(model, X_test, Y_test, category_names):
   """
     Function that evaluates a classification model based on accuracy, precision , recall and f1-score
     INPUT: model (ML model object) : Model to be evaluated
            X_test: (pandas dataframe): Dataframe Object containing test features
            y_test: (pandas dataframe): Dataframe Object containing test labels
            category_names (list): List containing category names
   """
   y_pred = model.predict(X_test)
   plot_f1_scores(Y_test,y_pred)
    

    
    


def save_model(model, model_filepath):
    """
        Function that saves a ML model as a pickle file
        INPUT: model(ML model): ML model to be saved
               model_filepath (str): String representation of the model to be saved
        OUTPUT: Saves the model as a pickle file
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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