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
    test = pd.read_csv("cleaned_test.csv")
    st.write("This will be the dataset used in this demo.")
    st.write(train)
    st.delay(1000)
    st.write("Here is a correlation graph between columns of this dataset.")
    fig = px.imshow(train.corr())
    st.plotly_chart(fig, use_container_width=True)
