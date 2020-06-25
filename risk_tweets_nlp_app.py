
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import socket
import json
import os

from risk_tweets_nlp_etl import lemmatize_document, STOPLIST

KERAS_MODEL = "my_keras_emb.h5"
MODEL_DIR = os.path.join(".","models")

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    query = request.json["data"]
    processed_query = [lemmatize_document(tweet, STOPLIST) for tweet in query]
    
    y_pred = model.predict_classes(processed_query)
    return(jsonify(y_pred.tolist()))
            
if __name__ == '__main__':
    saved_model = os.path.join(MODEL_DIR, KERAS_MODEL)
    model = keras.models.load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
    app.run(host='localhost', port=8080,debug=True)
