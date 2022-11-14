import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

def make_prediction(df, estimator, features_to_fit):

	# Create our target and labels
	X = df[features_to_fit]
	y = df["RainTomorrow"]
	#Identifying Numeric and categorical variables
	cat_vars = X.select_dtypes(include = ['object','category']).columns
	num_vars = X.select_dtypes(include = ['number'],exclude=['category']).columns

	# Create training and testing data sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
	random_state=43) 
	# Making the Pipeline
	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
	categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
	transformer = ColumnTransformer(transformers=[
	('num', numeric_transformer, num_vars),
	('cat', categorical_transformer, cat_vars)])
	pipe = Pipeline(steps=[
	("transformer", transformer),
	("clf", estimator)])
	# Fit the regressor with the full dataset to be used with predictions
	pipe.fit(X_train, y_train)

	# Do ten-fold cross-validation and compute our average accuracy
	cv = cross_val_score(pipe, X_test, y_test, cv=10)
	acc = cv.mean() * 100
	st.write(f"The accuracy is {round(acc, 2)}% (to 2 d.p.)")

	# Predict today's result
	X_new = df[features_to_fit]
	prediction = pipe.predict(X_new)

	# Return the predicted result
	return prediction
        
train = pd.read_csv("cleaned_train.csv")
train = train.drop(['Unnamed: 0'], axis = 1)
labels = None

option = st.sidebar.selectbox("Navigation",
("Starting Page", "Main Program (demo)")
)

if option == "Starting Page":
	st.title("Predictor - Eton College Environmental Hackathon")
	st.header("A machine-learning algorithm that uses past data and predicts possibility of natural disasters, which is made more prominent by climate change.")
	st.subheader("Made by Leo Sun (NPTL)")

if option == "Main Program (demo)":
	st.write("This will be the dataset used in this demo. Fear not about the value error raised at the bottom of this page -- it will disappear once parameters are placed.")
	st.write(train)
	st.write("Here is a correlation graph between columns of this dataset.")
	fig = px.imshow(train.corr())
	st.plotly_chart(fig, use_container_width=True)
	labels = st.multiselect("What are the labels required for the model? ", train.columns)
	if labels is not None:
		eee = make_prediction(train, LogisticRegression(), labels)
		pre_v_res = pd.DataFrame({
		"Location": train["Location"],
		"Actual Rain Status": train["RainTomorrow"],
		"Predicted Rain Status": eee})
		st.write("Here are the predicted results against the actual results.")
		st.write(pre_v_res)
		st.balloons()
		st.write("This is a confusion matrix of the Logistic Regressor used, which details the number of successful and failed predictions.")
		cm = confusion_matrix(train["RainTomorrow"], eee)
		fig2 = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), text_auto = True)
		st.plotly_chart(fig2, use_container_width=True)
