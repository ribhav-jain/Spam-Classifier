import pandas as pd
from flask import render_template, request
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app import app


@app.route("/train")
def train():
    df = pd.read_csv("./dataset/spam.csv", encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    df['label_as_num'] = df.label.map({"ham": 0, "spam": 1})
    label = df['label_as_num']
    text = df['text']

    count_vector = CountVectorizer()
    text = count_vector.fit_transform(text)
    joblib.dump(count_vector.vocabulary_, './saved_models/vocab.pkl')

    x_train, x_test, y_train, y_test = train_test_split(text, label, test_size=0.3, random_state=7)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, './saved_models/model.pkl')
    mse = mean_squared_error(y_test, y_pred)
    return "mse = " + str(mse) + "\ntraining completed."

@app.route("/")
@app.route("/index")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = 1
    model = open("./saved_models/model.pkl", 'rb')
    clf = joblib.load(model)
    vocab = open("./saved_models/vocab.pkl", 'rb')
    vocabulary = joblib.load(vocab)
    loaded_vectorizer = CountVectorizer(ngram_range=(size, size), min_df=1, vocabulary=vocabulary)
    loaded_vectorizer._validate_vocabulary()
    message = request.form['message']
    data = [message]
    vect = loaded_vectorizer.transform(data).toarray()
    prediction = clf.predict(vect)
    return render_template('result.html', prediction=prediction)
