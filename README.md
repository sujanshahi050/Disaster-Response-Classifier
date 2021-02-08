# Disaster Response Classifier

A web application that simulates the classification of texts into categories during a disaster which can be of a great addition to organizations and companies helping those in need during a disaster.

![](demo.gif)

#### Getting Started:

##### Required Libraries:

Apart from the libraries included in the Anaconda distribution for Python 3.6, following two libraries are included in the project.
1) nltk
2) sqlalchemy

##### Data

Data was downloaded from the [Figure8](https://appen.com/)

##### Installing 
 To clone this git repo:
 
 git clone https://github.com/sujanshahi050/Disaster-Response-Classifier.git

##### Running the Web App

Run the following commands to get the app running:
1) To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2) To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
3) Run the following command in the app's directory to run your web app. python run.py

4) Go to http://0.0.0.0:3001/

##### Files Description

     app/

        template/
    
            master.html - Main Page. 
            go.html -  Results Page.
            about.html- About Page
 
 
        run.py - flask app entry point.

    data/
 
        disaster_categories.csv - Disaster categories dataset.
        disaster_messages.csv - Disaster Messages dataset.
        process_data.py -  Python script to process and store data.
        DisasterResponse.db - The database with the merged and cleand data.

    models/

        train_classifier.py - Python Script that contains NLP and ML codes
        classifier.pkl - Pickle file that holds our ML model

    demo.gif - A small demo gif of the appliction.

##### Authors

Sujan Shahi

##### Acknowledgements

Udacity and Figure8


##### License

(https://opensource.org/licenses/MIT)

