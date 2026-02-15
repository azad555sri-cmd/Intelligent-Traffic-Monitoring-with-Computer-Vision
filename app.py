import streamlit as st
import cv2
import numpy as np
from tracker import EuclideanDistTracker

st.title("🚗 Interactive Vehicle Tracking Dashboard")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file is not None:
    # Save uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    # Tracker and Background Subtractor
    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    # Sidebar Controls
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Detection Threshold", 1, 255, 40)
    min_area = st.sidebar.slider("Minimum Contour Area", 10, 5000, 100)
    
    # ROI sliders (will update automatically based on video frame size)
    ret, test_frame = cap.read()
    if not ret:
        st.error("Unable to read video.")
    else:
        height, width, _ = test_frame.shape
        x1 = st.sidebar.slider("ROI x1", 0, width, 500)
        x2 = st.sidebar.slider("ROI x2", 0, width, 800)
        y1 = st.sidebar.slider("ROI y1", 0, height, 340)
        y2 = st.sidebar.slider("ROI y2", 0, height, 720)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    # Placeholders
    frame_placeholder = st.empty()
    roi_placeholder = st.empty()
    mask_placeholder = st.empty()
    count_placeholder = st.empty()

    unique_ids_set = set()

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.info("End of video")
            break

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        # Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        # Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            unique_ids_set.add(id)
            cv2.putText(roi, str(id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Convert for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Update Streamlit placeholders
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Full Frame")
        roi_placeholder.image(roi_rgb, channels="RGB", caption="Region of Interest (ROI)")
        mask_placeholder.image(mask_rgb, channels="RGB", caption="Mask")
        count_placeholder.metric("Total Vehicles Detected", len(unique_ids_set))

    cap.release()
