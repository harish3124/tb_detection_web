# Step 1: Load the CSV file into a Pandas DataFrame
import pandas as pd
df = pd.read_csv('Features Extraction File.csv')
import numpy as np
# Step 2: Separate the label column from the rest of the columns and convert to numpy array
from sklearn.preprocessing import LabelEncoder
X = df.drop(['filename', 'label'], axis=1).to_numpy()
le=LabelEncoder()
y = le.fit_transform(df['label'])

# Step 4: Normalize the input data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Define the deep learning model
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

# Step 7: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Step 9: Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()  # Convert the predicted probabilities to binary predictions

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, y_pred))
report = classification_report(y_test, y_pred)
print(report)
import pickle

with open('deepL_model_pickle', 'wb') as f:
    pickle.dump(model, f)
