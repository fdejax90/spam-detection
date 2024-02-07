import requests
import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8080/predict/"


def process(sms: str, server_url: str):
    r = requests.post(server_url, json={"sms": sms}, timeout=8000)
    return r


# construct UI layout
st.title("SMS spam detection")

# description and instructions
st.write(
    """Effectively identify and classify SMS messages as either 'spam' or 'not spam'.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at [docs](http://localhost:8080/docs) for FastAPI documentation."""
)

input_sms = st.text_input("Enter your SMS:")

if st.button("Spam detector"):
    if input_sms:
        sms = process(input_sms, backend)
        st.header("Answer")
        st.write(sms.json())

    else:
        # handle case with no sms
        st.write("Write a SMS!")
