from ultralytics import YOLO
import streamlit as st
import cv2 as cv
import tempfile

# Load YOLO model
model = YOLO("C:/Users/shana/Downloads/yolov8n.pt")

st.title("YOLO Object Detection App")

# Select mode
mode = st.radio("Choose input type:", ("Image", "Video", "Webcam"))

# ------------------ IMAGE MODE ------------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpeg', 'webp', 'heic', 'png', 'tif', 'pfm', 'tiff','bmp', 'jpg', 'dng', 'mpo'])
    
    if uploaded_file is not None:
        # Save temporarily
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            img_path = tmp.name

        # Run YOLO
        results = model(img_path)
        res_plotted = results[0].plot()

        # Show result
        st.image(res_plotted, channels="BGR")
    else:
        st.warning("Please upload an image file.")

# ------------------ VIDEO MODE ------------------
elif mode == "Video":
    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["ts", "wmv", "mpg", "mov", "asf", "mp4", "m4v", "webm", "mkv", "avi", "mpeg", "gif"],
    )
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()
    else:
        st.warning("Please upload a video file.")

# ------------------ WEBCAM MODE ------------------
elif mode == "Webcam":
    run = st.checkbox("Start Webcam")
    stframe = st.empty()

    if run:
        cap = cv.VideoCapture(0)  # 0 = default webcam
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible")
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()
