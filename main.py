import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots

options = st.sidebar.selectbox(
    "Contents",
    ("Starting Page", "Main Program (demo)")
)

if options == "Starting Page":
    st.title("Predictor - Eton College Environmental Hackathon")
    st.header("A machine-learning algorithm that uses past data and predicts possibility of natural disasters, which is made more prominent by climate change.")
    st.subheader("Made by Leo Sun (NPTL)")
if options == "Main Program (demo)":
    train = pd.read_csv("cleaned_train.csv")
    train = train.drop(['Unnamed: 0'], axis = 1)
    st.write("This will be the dataset used in this demo.")
    st.write(train)
    st.write("Here is a correlation graph between columns of this dataset.")
    fig = px.imshow(train.corr())
    st.plotly_chart(fig, use_container_width=True)
    labels = st.text_input(label = "Please enter the labels needed for the model to run. (Please seperate words with a single space.)").split()
    to_predict = st.text_input(label = "Which label do you want to predict?")
    
    def make_prediction(df, estimator, features_to_fit, to_predict):
    
    # Create our target and labels
        X = df[features_to_fit]
        y = df[to_predict]
    
    # Create training and testing data sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
            random_state=43) 
    
    # Fit the regressor with the full dataset to be used with predictions
        estimator.fit(X, y)
    
    # Do ten-fold cross-validation and compute our average accuracy
        cv = cross_val_score(estimator, X_test, y_test, cv=10)
        print('Accuracy:', cv.mean())

    # Predict today's closing price
        X_new = df_today[features_to_fit]
        prediction = estimator.predict(X_new)
    
    # Return the predicted closing price
        return prediction
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    print('Predicted Results: %.2f\n' % make_prediction(train, pipe))
