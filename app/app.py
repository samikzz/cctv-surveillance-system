import streamlit as st
import os
import cv2
import tempfile
import requests

# Directory to save the uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("CCTV Camera Surveillance")

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
            predictions = response.json()["predictions"]

            # Open the video file and create a temporary file to save the annotated video
            cap = cv2.VideoCapture(video_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            out_path = temp_file.name

            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

            segment_duration = 5  # Duration of each segment in seconds
            current_time = 0  # To keep track of the video time

            for prediction in predictions:
                start_time = prediction["start_time"]
                end_time = prediction["end_time"]
                label = prediction["prediction"]

                # Set the video to the start of the prediction segment
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

                while current_time < end_time:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Overlay the prediction on the video
                    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    out.write(frame)

                    # Increment the current time based on FPS
                    current_time += 1 / fps

            cap.release()
            out.release()
            temp_file.close()

            st.session_state["download_ready"] = out_path

            # Render the download button if the video is ready to be downloaded
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="video/avi"
                )
