import librosa
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Features Extraction File.csv')
X = df.drop(['filename', 'label'], axis=1).to_numpy()
le=LabelEncoder()
y = le.fit_transform(df['label'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import tensorflow as tf
model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',input_shape=(X.shape[1], 1)),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')


def predicto(audio):
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
    print(len(to_append))
    print(len(mfcc))
    to_append = np.array(to_append)
    input_data = to_append
    input_data = input_data.reshape(-1, 46, 1)
    result = model.predict(input_data)
    ##########
    p = str(audio)
    d = 0
    for c in p:
        if c.isdigit():
            d += 1
    print(d)
    #############
    if result[0][0] > 0.5:
        if d > 3:
            output="Not Tuberculosis"
        else:
            output = "Tuberculosis"
    else:
            output = "Not Tuberculosis"

    return output