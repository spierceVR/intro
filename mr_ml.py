import pandas as pd
from IPython.display import display

#import csv as a pd dataframe
mr = pd.read_csv('./input/mushrooms.csv')
display(mr.head())

#copy dataframe
df = mr.copy()


# binaryencoding on class column
df['class'] = df['class'].map({'p': 0, 'e': 1})
display(df.head())

#split df into training and validating datasets (70/30)
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

# for normalizing numerical data in the tutorial, I dont think I need this
# max_ = df_train.max(axis=0)
# min_ = df_train.min(axis=0)


# convert cat data in every column to numerical (OneHotEncoding?)
from sklearn import preprocessing
lab_encoder = preprocessing.LabelEncoder()

for col in df.columns:
    if(col == 'class'):
        df[col] = lab_encoder.fit_transform(df[col])



from tensorflow import keras
from tensorflow.keras import layers



X_train = df_train.drop('class', axis=1)
X_valid = df_valid.drop('class', axis=1)
y_train = df_train['class']
y_valid = df_valid['class']

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[22]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)


history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))