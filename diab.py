import pandas as pd

train_df = pd.read_csv('C:/Users/c-jatin.lalwani/Downloads/Deep-Learning-in-Keras-Tutorial-master/data/diabetes_data.csv')

train_X = train_df.drop(columns=['diabetes'])
print(train_X.values)

train_Y = train_df['diabetes']
print(train_Y.values)

test_X = train_df.drop(columns=['diabetes'])[-5:]

test_Y = train_df['diabetes'][-5:]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

model = Sequential()

n_cols = train_X.shape[1]
print(train_X.shape)
print(train_Y.values.shape)

model.add(Dense(200,input_shape=(n_cols,)))
model.add(Dense(1000,activation="relu"))
model.add(Dense(1000,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_X.values,train_Y.values,epochs=30)

test_y_predictions = model.predict(test_X.values)

print(test_Y.values)
print(test_y_predictions)

