from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
import pickle

# loading the model
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        predictions = clf.predict(vect)

    return render_template('result.html', prediction = predictions)

if __name__ == '__main__':
    app.run(debug=True)