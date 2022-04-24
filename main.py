from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import tensorflow as tf
from IPython.display import display

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=512, input_shape=[22]),
    layers.Activation('relu'),
    layers.Dense(units=512),
    layers.Activation('relu'),
    layers.Dense(units=512),
    layers.Activation('relu'),
    layers.Dense(units=1),
    # input_shape is the amount of columns in a tabular dataset
    # Image data, for instance, might need three dimensions: [height, width, channels].
])
# w, b = model.weights

model.compile(
    optimizer="adam",
    loss="mae",
)

# print("Weights\n{}\n\nBias\n{}".format(w, b))

import pandas as pd

mr = pd.read_csv('./input/mushrooms.csv')
print(mr.head())

# Create training and validation splits
df_train = mr.sample(frac=0.7, random_state=0)
df_valid = mr.drop(df_train.index)
display(df_train.head())

# Scale to [0, 1]
# max_ = df_train.max(axis=0)
# min_ = df_train.min(axis=0)
# df_train = (df_train - min_) / (max_ - min_)
# df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('class', axis=1)
X_valid = df_valid.drop('class', axis=1)
y_train = df_train['class']
y_valid = df_valid['class']

display(X_train.head())
display(y_train.head())