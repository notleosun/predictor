import streamlit as st

options = st.sidebar.selectbox(
    "Contents",
    ("Starting Page", "Demo page (used in pitching video only)", "Main Program")
)
if options == "Starting Page":
    st.title("RainTomorrow - Eton College Environmental Hackathon")
    st.header("A machine-learning algorithm that uses past weather data to try and predict extreme downpour, which is made more prominent by climate change.")
    st.subheader("Made by Leo Sun (NPTL)")
if options == "Main Program":
    st.file_uploader(label = "Upload your .csv file here: ")
