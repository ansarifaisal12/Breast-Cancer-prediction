import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import seaborn as sns

# Load the breast cancer dataset from scikit-learn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Display results
st.title("Breast Cancer Classification App")

st.write("""
## Classify breast cancer as benign or malignant
""")

# Show the dataset
if st.checkbox('Show dataset'):
    st.write(X.head())

# Show the statistics of the dataset
if st.checkbox('Show data description'):
    st.write(X.describe())

# Show the model accuracy
st.write('Model Accuracy:', accuracy_score(y_test, y_pred))

# Show the classification report
st.write('Classification Report:')
st.write(classification_report(y_test, y_pred))

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    input_features = {}
    for feature in X.columns:
        input_features[feature] = st.sidebar.slider(f'Enter {feature}', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    return pd.DataFrame([input_features])

input_df = user_input_features()

# Predict and display the result
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.subheader('Prediction')
st.write('Predicted Class:', data.target_names[prediction][0])
st.write('Prediction Probability:', prediction_proba)

st.write("""
## About Dataset
The Breast Cancer dataset is a classic and publicly available dataset in scikit-learn. It contains measurements of breast cancer tumors, and the task is to predict whether the tumor is benign or malignant based on these measurements.
""")


