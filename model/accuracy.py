import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from json import dumps


def gbc():
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train, y_train)
    y_pred = gbc.predict(x_test)
    ac_gbc = accuracy_score(y_pred, y_test)
    ac_gbc = ac_gbc * 100
    return ac_gbc


def rf():
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    ac_rf = accuracy_score(y_pred, y_test)
    ac_rf = ac_rf * 100
    return ac_rf


def cnn():
    cnn = RandomForestClassifier()
    cnn.fit(x_train, y_train)
    y_pred = cnn.predict(x_test)
    ac_cnn = accuracy_score(y_pred, y_test)
    ac_cnn = ac_cnn * 100
    return ac_cnn


def dts():
    dts = DecisionTreeClassifier()
    dts.fit(x_train, y_train)
    y_pred = dts.predict(x_test)
    ac_dts = accuracy_score(y_pred, y_test)
    ac_dts = ac_dts * 100
    return ac_dts


def acc():
    global x_train
    global x_test
    global y_train
    global y_test

    df = pd.read_csv("model/Features Extraction File.csv")
    df.head()
    df = df[
        [
            "spectral_centroid",
            "spectral_bandwidth",
            "rolloff",
            "mfcc1",
            "mfcc2",
            "mfcc3",
            "mfcc5",
            "mfcc6",
            "mfcc8",
            "mfcc12",
            "mfcc14",
            "mfcc21",
            "mfcc30",
            "mfcc32",
            "mfcc34",
            "mfcc36",
            "label",
        ]
    ]
    df.head()
    x = df.drop(["label"], axis=1)
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=100
    )

    return dumps({"gbc": gbc(), "rf": rf(), "cnn": cnn(), "dts": dts()})
