import json
import plotly
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
import sqlite3

app = Flask(__name__)
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# Load data
conn = sqlite3.connect('web_application\\data\\DisasterResponse.db')
query = "SELECT * FROM DisasterResponse"
df = pd.read_sql_query(query, conn)

# Load the model for inference
hub_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=True)

model1 = tf.keras.models.load_model('best_model1.h5', custom_objects={'KerasLayer': hub.KerasLayer, 'f1_score': f1_score})
model2 = tf.keras.models.load_model('best_model2.h5', custom_objects={'KerasLayer': hub.KerasLayer, 'f1_score': f1_score})
model3 = tf.keras.models.load_model('best_model3.h5', custom_objects={'KerasLayer': hub.KerasLayer, 'f1_score': f1_score})

# Index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    X = df['message']
    Y = df.iloc[:,4:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    # Data for bar plot
    category_pct = Y_train.mean().sort_values(ascending= False)
    category = category_pct.index.str.replace('_', ' ')

    # Data for score distribution
    score_dist = df.iloc[:, 4:].sum().sort_values(ascending=False)

    # Data for message counts by genre
    message_counts_by_genre = df.groupby('genre').count()['message']

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category,
                    y=category_pct
                )
            ],
            'layout': {
                'title': {
                    'text': 'Proportions of categories',
                    'font': {'size': 18, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                },
                'yaxis': {
                    'title': {
                        'text':"Percentage",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickformat': ',.0%',
                },
                'xaxis': {
                    'title': {
                        'text':"Category",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickangle': 45
                },
                'height': 800, 
                'width': 1100,
                'margin': {
                    'l': 150, 
                    'r': 100,
                    'b': 200, 
                    't': 100, 
                    'pad': 4 
                }
            }
        },
        {
            'data': [
                Bar(
                    x=score_dist.index,
                    y=score_dist.values
                )
            ],
            'layout': {
                'title': 'Distribution of Scores',
                'xaxis': {'title': 'Category'},
                'yaxis': {'title': 'Score'}
            }
        },
        {
            'data': [
                Bar(
                    x=message_counts_by_genre.index,
                    y=message_counts_by_genre.values
                )
            ],
            'layout': {
                'title': 'Distribution of Messages by Genre',
                'xaxis': {'title': 'Genre'},
                'yaxis': {'title': 'Number of Messages'}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Prepare the input for the model
    input_data = [query]

    # Make predictions
    predicted_output1 = model1.predict(input_data)
    predicted_output2 = model2.predict(input_data)
    predicted_output3 = model3.predict(input_data)

    classification_labels1 = np.where(predicted_output1 > 0.4, 1, 0)
    classification_labels2 = np.where(predicted_output2 > 0.45, 1, 0)
    classification_labels3 = np.where(predicted_output3 > 0.5, 1, 0)

    # Concatenate the results
    horizontal_concatenation = np.concatenate((classification_labels3, classification_labels2, classification_labels1), axis=1)
    classification_results = dict(zip(df.columns[4:], horizontal_concatenation[0]))

    # Render the go.html
    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
