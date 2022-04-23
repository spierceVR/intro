from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
# model = keras.Sequential([
#     layers.Dense(units=1, input_shape=[3]) 
#     # input_shape is the amount of columns in a tabular dataset
#     # Image data, for instance, might need three dimensions: [height, width, channels].
# ])


import pandas as pd

mr = pd.read_csv('./input/mushrooms.csv')
print(mr.head())

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
x = np.linspace(0, 10, 100)
y = 4 + 2 * np.sin(2 * x)

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()