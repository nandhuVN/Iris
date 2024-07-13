import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load Iris dataset for feature names
iris = load_iris()

st.title("Iris Flower Species Prediction")
st.write("This is a simple Streamlit app to predict the species of Iris flower based on its features.")

# Sidebar inputs for features
sepal_length = st.sidebar.slider('Sepal Length', float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
petal_length = st.sidebar.slider('Petal Length', float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
petal_width = st.sidebar.slider('Petal Width', float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))

# Predict button
if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    st.write(f"Predicted species: {iris.target_names[prediction][0]}")
    st.write(f"Prediction probabilities: {prediction_proba}")

