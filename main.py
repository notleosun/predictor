import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots

options = st.sidebar.selectbox(
    "Contents",
    ("Starting Page", "Demo page (used in pitching video only)", "Main Program")
)
if options == "Starting Page":
    st.title("Predictor - Eton College Environmental Hackathon")
    st.header("A machine-learning algorithm that uses past data and predicts possibility of natural disasters, which is made more prominent by climate change.")
    st.subheader("Made by Leo Sun (NPTL)")
if options == "Main Program":
    uploaded_file = st.file_uploader(label = "Upload your .csv file here: ")
    print(uploaded_file)
    fig = px.imshow(uploaded_file.corr())
    st.plotly_chart(fig, use_container_width=True)
