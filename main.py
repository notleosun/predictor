import streamlit as st
import pandas as pd

options = st.sidebar.selectbox(
    "Contents",
    ("Starting Page", "Demo page (used in pitching video only)", "Main Program")
)
if options == "Starting Page":
    st.title("Predictor - Eton College Environmental Hackathon")
    st.header("A machine-learning algorithm that uses past data and predicts possibility of natural disasters, which is made more prominent by climate change.")
    st.subheader("Made by Leo Sun (NPTL)")
if options == "Main Program":
    dataset = st.file_uploader(label = "Upload your .csv file here: ")
    dataset = dataset.read()
    st.write(dataset)
