import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()

print(train_images)
print("----------------------------------------------")
print(train_labels)
print("=============================================")
print(test_images)
print(test_labels)

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #This is done to shrink the data as each pixel ranges from 0 to 255 we now reduced the range to 0 to 1
# train_images = train_images/255
# test_images = test_images/255       

# model = keras.Sequential([
# 	keras.layers.Flatten(input_shape=(28,28)),
# 	keras.layers.Dense(128,activation="relu"),
# 	keras.layers.Dense(10,activation="softmax")
# 	])

# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# model.fit(train_images,train_labels,epochs=5)

# prediction = model.predict(test_images)
