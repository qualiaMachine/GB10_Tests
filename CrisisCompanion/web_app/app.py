import streamlit as st
import requests

st.title("Qwen Chatbot")

# Text input
user_input = st.text_input("You:", "")

# Send button
if st.button("Send") and user_input.strip():
    try:
        # Call FastAPI backend
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            headers={"Content-Type": "application/json"},
            json={"input": user_input}
        )
        bot_message = response.json().get("response", "")
        st.text_area("Bot:", value=bot_message, height=200)
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
