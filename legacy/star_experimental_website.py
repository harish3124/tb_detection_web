from cProfile import run

import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from flask import *
import tensorflow as tf
import IPython.display as ipd

app = Flask(__name__)

df = pd.read_csv('Features Extraction File.csv')
df.head()
df = df[['spectral_centroid', 'spectral_bandwidth', 'rolloff', 'mfcc1', 'mfcc2',
         'mfcc3', 'mfcc5', 'mfcc6', 'mfcc8', 'mfcc12', 'mfcc14', 'mfcc21',
         'mfcc30', 'mfcc32', 'mfcc34', 'mfcc36', 'label']]
df.head()
x = df.drop(['label'], axis=1)
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=100)

#                     deep learning model related below
data = df
features = data.iloc[:, 1:-1]
labels = data.iloc[:, -1]
labels = (labels == "covid").astype(int)
train_features, test_features, train_labels, test_labels = \
train_test_split(features, labels, test_size=0.2)
train_features = train_features.values.reshape(-1, features.shape[1], 1)
test_features = test_features.values.reshape(-1, features.shape[1], 1)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/view')
def view():
    dataset = pd.read_csv('Features Extraction File.csv')
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/model', methods=["POST", "GET"])
def model():
    if request.method == "POST":
        global x_train, x_test, y_train, y_test
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg="Choose an algorithm")
        elif s == 2:
            gbc = GradientBoostingClassifier()
            gbc.fit(x_train, y_train)
            y_pred = gbc.predict(x_test)
            ac_gbc = accuracy_score(y_pred, y_test)
            ac_gbc = ac_gbc * 100
            msg = "The acuuracy obtained by Gradient Boosting Classifier is  " + str(ac_gbc) + str("%")
            return render_template("model.html", msg=msg)
        elif s == 3:
            dts = DecisionTreeClassifier()
            dts.fit(x_train, y_train)
            y_pred = dts.predict(x_test)
            ac_dts = accuracy_score(y_pred, y_test)
            ac_dts = ac_dts * 100
            msg = "The accuracy obtained by Decision tree Classifier " + str(ac_dts) + str('%')
            return render_template("model.html", msg=msg)
        elif s == 4:
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            ac_rf = accuracy_score(y_pred, y_test)
            ac_rf = ac_rf * 100
            msg = "The accuracy obtained by Randomforest Classifier is " + str(ac_rf) + str('%')
            return render_template("model.html", msg=msg)
        elif s == 5:
            global deep_model
            deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = deep_model.fit(train_features, train_labels, epochs=100, validation_split=0.2, verbose=0)
            test_loss, test_acc = deep_model.evaluate(test_features, test_labels)
            test_acc*=100
            msg = "The accuracy obtained by Convolutional Neural Network is " + str(test_acc) + str('%')
            return render_template("model.html", msg=msg)

    return render_template("model.html")


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    global x_train, x_test, y_train, y_test
    if request.method == "POST":
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])
        f12 = float(request.form['f12'])
        f13 = float(request.form['f13'])
        f14 = float(request.form['f14'])
        f15 = float(request.form['f15'])
        f16 = float(request.form['f16'])

        lee = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]
        print(lee)

        import pickle
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        result = model.predict([lee])
        print(result)
        result = result[0]

        if result == 'covid':
            msg = 'The person has Tuberculosis, please consult a doctor.'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'This person does not have Tuberculosis.'
            return render_template('prediction.html', msg=msg)

    return render_template('prediction.html')


@app.route("/upload", methods=["GET", "POST"])
def up():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            from stackerfile import website_stacker_classifier
            transcript=website_stacker_classifier(file)

    return render_template('upload.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True)
