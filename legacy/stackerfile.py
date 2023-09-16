import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load data into a pandas dataframe
df = pd.read_csv("Features Extraction File.csv")

# Split data into features (X) and labels (y)
X = df.drop(["filename", "label"], axis=1)
y = df["label"]


# Train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5)

# Train the Gradient Boosting Machine classifier
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5)

# Initialize the StackingClassifier
clf = StackingClassifier(estimators=[("rf", rf),("gbm", gbm),("dt",dt)], final_estimator=LogisticRegression())
clf.fit(X, y)
# # Use cross_val_score to get a more robust estimate of performance
# rf_scores = cross_val_score(rf, X, y, cv=5)
# gbm_scores = cross_val_score(gbm, X, y, cv=5)
# dt_scores = cross_val_score(dt, X, y, cv=5)
# stacking_scores = cross_val_score(clf, X, y, cv=5)
#
# # Calculate mean accuracy and standard deviation for each classifier
# rf_mean_acc = rf_scores.mean()
# rf_std_acc = rf_scores.std()
# gbm_mean_acc = gbm_scores.mean()
# gbm_std_acc = gbm_scores.std()
# dt_mean_acc = dt_scores.mean()
# dt_std_acc = dt_scores.std()
# stacking_mean_acc = stacking_scores.mean()
# stacking_std_acc = stacking_scores.std()
#
# print("Individual classifiers mean accuracy scores and standard deviations:")
# print("Random Forest classifier mean accuracy:", rf_mean_acc, "standard deviation:", rf_std_acc)
# print("Gradient Boosting classifier mean accuracy:", gbm_mean_acc, "standard deviation:", gbm_std_acc)
# print("Decision Tree classifier mean accuracy:", dt_mean_acc, "standard deviation:", dt_std_acc)
# print("Stacking Classifier mean accuracy:", stacking_mean_acc, "standard deviation:", stacking_std_acc)

# ##############################################################################################################
# chroma_stft = 0.54650414
# rmse = 0.030731075
# spectral_centroid = 1756.939071
# spectral_bandwidth = 1920.931232
# rolloff = 3803.399291
# zero_crossing_rate = 0.069694795
# mfcc1 = -391.1262817
# mfcc2 = 84.53215027
# mfcc3 = -7.05690527
# mfcc4 = 33.71518707
# mfcc5 = 20.2015686
# mfcc6 = 20.77123642
# mfcc7 = 3.401362419
# mfcc8 = 4.628528595
# mfcc9 = 2.157515049
# mfcc10 = 11.39009571
# mfcc11 = 1.224411964
# mfcc12 = 0.403367341
# mfcc13 = 5.491996288
# mfcc14 = 3.340778351
# mfcc15 = 1.814354539
# mfcc16 = -2.554027081
# mfcc17 = -4.736023903
# mfcc18 = -2.298579454
# mfcc19 = -2.910229445
# mfcc20 = 0.922017395
# mfcc21 = 0.654921651
# mfcc22 = 0.721835077
# mfcc23 = 2.065487862
# mfcc24 = 2.385349751
# mfcc25 = 1.406109095
# mfcc26 = 1.472823143
# mfcc27 = -1.080055356
# mfcc28 = -1.410896897
# mfcc29 = 0.06093641
# mfcc30 = -2.055373669
# mfcc31 = -1.475679517
# mfcc32 = -1.142547011
# mfcc33 = -0.036875393
# mfcc34 = -0.463881433
# mfcc35 = -1.210339785
# mfcc36 = -2.041661978
# mfcc37 = -0.094270594
# mfcc38 = 0.005632492
# mfcc39 = -0.393657327
# mfcc40 = -0.438715696
# ###############################################################################

def website_stacker_classifier(audio):
    y, sr = librosa.load(audio, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    to_append = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
                 np.mean(zcr)]
    for e in mfcc:
        to_append.append(np.mean(e))
    to_append = np.array(to_append)
    to_append = to_append.reshape(1, -1)
    pred_class = clf.predict(to_append)
    if pred_class !="not_covid":
        transcript = "Tuberculosis"
    else:
        transcript = "Not Tuberculosis"
    return transcript



# X_input = pd.DataFrame({"chroma_stft": [chroma_stft],
#                         "rmse": [rmse],
#                         "spectral_centroid": [spectral_centroid],
#                         "spectral_bandwidth": [spectral_bandwidth],
#                         "rolloff": [rolloff],
#                         "zero_crossing_rate": [zero_crossing_rate],
#                         "mfcc1": [mfcc1],
#                         "mfcc2": [mfcc2],
#                         "mfcc3": [mfcc3],
#                         "mfcc4": [mfcc4],
#                         "mfcc5": [mfcc5],
#                         "mfcc6": [mfcc6],
#                         "mfcc7": [mfcc7],
#                         "mfcc8": [mfcc8],
#                         "mfcc9": [mfcc9],
#                         "mfcc10": [mfcc10],
#                         "mfcc11": [mfcc11],
#                         "mfcc12": [mfcc12],
#                         "mfcc13": [mfcc13],
#                         "mfcc14": [mfcc14],
#                         "mfcc15": [mfcc15],
#                         "mfcc16": [mfcc16],
#                         "mfcc17": [mfcc17],
#                         "mfcc18": [mfcc18],
#                         "mfcc19": [mfcc19],
#                         "mfcc20": [mfcc20],
#                         "mfcc21": [mfcc21],
#                         "mfcc22": [mfcc22],
#                         "mfcc23": [mfcc23],
#                         "mfcc24": [mfcc24],
#                         "mfcc25": [mfcc25],
#                         "mfcc26": [mfcc26],
#                         "mfcc27": [mfcc27],
#                         "mfcc28": [mfcc28],
#                         "mfcc29": [mfcc29],
#                         "mfcc30": [mfcc30],
#                         "mfcc31": [mfcc31],
#                         "mfcc32": [mfcc32],
#                         "mfcc33": [mfcc33],
#                         "mfcc34": [mfcc34],
#                         "mfcc35": [mfcc35],
#                         "mfcc36": [mfcc36],
#                         "mfcc37": [mfcc37],
#                         "mfcc38": [mfcc38],
#                         "mfcc39": [mfcc39],
#                         "mfcc40": [mfcc40]})
# print(X_input)
# pred_class = clf.predict(X_input)
#
# print(pred_class)




