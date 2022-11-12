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
    train = pd.read_csv("cleaned_train")
    test = pd.read_csv("cleaned_test")
    st.write("This will be the dataset used in this demo.")
    st.write(train)
    fig = px.imshow(uploaded_file.corr())
    st.plotly_chart(fig, use_container_width=True)
