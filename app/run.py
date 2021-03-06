import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
        Function that takes a text(str) as an input, cleans it and returns a lemmatized list of tokens(words)
        
        INPUT: text (str) : Text to be cleaned and tokenized and lemmatized
        OUTPUT: clean_tokens(list): List of clean tokens extracted from the text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Distribution of categoires
    categories = df.iloc[:,4:].sum().sort_values().reset_index()
    categories.columns = ['category','count']
    categories_labels = categories['category'].values.tolist()
    categories_values = categories['count'].values.tolist()
    
    # 
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [  
        # Graph data for distribution of messages genres
        {'data':[
            Bar(
                x = genre_names,
                y = genre_counts     
            )
        
        ],
         'layout':{'title': 'Distribution of Messages Genre',
                   'yaxis':{
                       'title': "Count"
                   },
                   'xaxis':{
                        'title': "genre"
                   }
            }
        },
        
        # Graph data for Messages Categories Distribution
        {
            'data': [
                Bar(
                    x=categories_labels,
                    y=categories_values,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

@app.route('/about')
def about():
    return render_template(
        'about.html'
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()