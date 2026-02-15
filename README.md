AI-Powered Vehicle Detection & Tracking

A real-time AI-driven system for detecting, tracking, and analyzing vehicles using computer vision and Streamlit.

🚀 Project Overview

This project leverages computer vision (OpenCV) and Euclidean Distance-based object tracking to monitor traffic in real-time. It features an interactive Streamlit dashboard where users can adjust parameters like region-of-interest (ROI), detection thresholds, and minimum contour area. The system counts vehicles passing through a defined area and visualizes the results live.


Key Features:

- Real-time vehicle detection using background subtraction (MOG2).

- Euclidean Distance-based object tracking with unique IDs.

- Interactive ROI selection and detection threshold sliders.

- Live vehicle count display.

- Visual outputs for Full Frame, ROI, and Mask.

- Streamlit-based browser dashboard for easy access and interaction.


💻 Tech Stack

- Language: Python 3.x

- Libraries: OpenCV, NumPy, Streamlit

- Tracking: Custom Euclidean Distance Tracker

- Visualization: Real-time video stream in Streamlit


🛠 Usage

- Adjust ROI sliders to select the area of interest.

- Adjust Detection Threshold and Minimum Contour Area to optimize detection.

- View real-time video, ROI, mask, and vehicle count in the dashboard.
