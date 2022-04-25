import pandas as pd
from sklearn import preprocessing
from tensorflow import keras

### Setup preprocessing pipeline
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

dataToFit = pd.read_csv("./input/mushrooms.csv")
dataToFit.pop('class')
preprocessor = preprocessor.fit(dataToFit)


# Create dataframe of input data for prediction
X_predict = pd.DataFrame(columns=["class", "cap-shape", "cap-surface", "cap-color","bruises",
    "odor","gill-attachment","gill-spacing","gill-size",
    "gill-color","stalk-shape","stalk-root",
    "stalk-surface-above-ring","stalk-surface-below-ring",
    "stalk-color-above-ring","stalk-color-below-ring",
    "veil-type","veil-color","ring-number","ring-type",
    "spore-print-color","population","habitat"],
    index = [0],
    # mushroom info (this row does not appear in the database used for training/testing): p  x   s	n	t	p	f	c	n	k	e	e	s	s	w	w	p	w	o	p	k	s	u
    data = [ ["p", "x", "s", "n", "t", "p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","s","u"] ])

# Perform preprocessing on prediction input
X_predict.pop('class')
X_predict = preprocessor.transform(X_predict)
X_predict = X_predict.toarray()

# Load pre-trained model
model =  keras.models.load_model('saved_model/my_model')

# Make prediction and show result
prediction = model.predict(X_predict)
prediction_class = lambda x : "edible" if (x >= 0.5) else "poisonous"
print(prediction[0], "  classification: ", prediction_class(prediction[0][0])) 