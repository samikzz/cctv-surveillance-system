import streamlit as st
import os
import requests

# Directory to save the uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Video Classification Model")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video locally
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)

    if st.button("Classify Video"):
        # Send the video path to FastAPI for classification
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            json={"video_path": video_path},
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error: Could not classify the video.")
