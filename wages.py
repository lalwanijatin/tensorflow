import pandas as pd

train_df = pd.read_csv('C:/Users/c-jatin.lalwani/Downloads/Deep-Learning-in-Keras-Tutorial-master/data/hourly_wages_data.csv')

train_X = train_df.drop(columns=['wage_per_hour'])[:-5]

train_Y = train_df['wage_per_hour'][:-5]

test_X = train_df.drop(columns=['wage_per_hour'])[-5:]

test_Y = train_df['wage_per_hour'][-5:]

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

n_cols = train_X.shape[1]

model.add(Dense(10,activation="relu",input_shape=(n_cols,)))
model.add(Dense(1000,activation="relu"))
model.add(Dense(1000,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])

model.fit(train_X,train_Y,validation_split=0.2,epochs=30)

test_y_predictions = model.predict(test_X)

print(test_Y)
print(test_y_predictions)

