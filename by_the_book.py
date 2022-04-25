# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


# Load the dataset
import pandas as pd
shrooms = pd.read_csv('./input/mushrooms.csv')


# Binary encoding on class column
shrooms['class'] = shrooms['class'].map({'p': 0, 'e': 1})
from IPython.display import display
display(shrooms.head())


# Create input and output sets
X = shrooms.copy()
y = X.pop('class')


# Set up preprocessing pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

features_cat = [
    "cap-shape", "cap-surface", "cap-color","bruises",
    "odor","gill-attachment","gill-spacing","gill-size",
    "gill-color","stalk-shape","stalk-root",
    "stalk-surface-above-ring","stalk-surface-below-ring",
    "stalk-color-above-ring","stalk-color-below-ring",
    "veil-type","veil-color","ring-number","ring-type",
    "spore-print-color","population","habitat"
]

transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_cat, features_cat)
)

# Stratify - make sure classes are evenly represented across splits
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)


# Perform preprocessing on the training and validation data
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_train = X_train.toarray()
X_valid = X_valid.toarray()


# Setup ML model
from tensorflow import keras
from tensorflow.keras import layers, Sequential, callbacks

input_shape = [X_train.shape[1]]
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),    
    layers.BatchNormalization(),
    layers.Dropout(0.3),
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


# Train for supplied amount of epochs
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
    # verbose=0, # hide the output because we have so many epochs
)


# Plot results of model training
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
history_df.loc[0:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max())
    )
    
plt.show()

# model.save('saved_model/my_model')