# rppg-project
Blood Pressure Prediction Using Facial Features & rPPG Signals
 Project Overview
This project aims to estimate systolic and diastolic blood pressure using facial images/videos captured from a webcam. It utilizes remote photoplethysmography (rPPG) for non-contact feature extraction and applies deep learning models (CNN, LSTM) for real-time blood pressure estimation and monitoring.

🔧 Features
Real-time face and eye detection using OpenCV

rPPG signal extraction for pulse-based blood pressure estimation

Deep learning models for blood pressure prediction


Pyqt6-based GUI for user-friendly interaction

Executable application created using InstallForge

🛠 Installation
Prerequisites
Ensure you have the following installed:

Python 

OpenCV (pip install opencv-python)

TensorFlow/Keras (pip install tensorflow)

PyTorch (optional, if using alternative models)

Flask/Django (if deploying as a web API)

Pyqt6 (for the GUI)

Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
python app.py  # If using Flask/Django
python gui.py   # If using Tkinter
📊 Dataset


Custom dataset collected using a infrared camera and BP monitor

Features include facial images, rPPG signals, and BP readings

🧠 Model Architecture
CNN for facial feature extraction

LSTM for temporal signal analysis (rPPG processing)

MLP for final blood pressure estimatio

Data augmentation and preprocessing techniques applied

🖥 Usage
Webcam Mode: Detects face, extracts rPPG, and predicts BP in real time

Image Input Mode: Allows users to upload facial images for BP estimation

GUI Mode: A user-friendly Pyqt6 interface for easy operation

📜 License
This project is licensed under the MIT License.

✉ Contact
For any queries, feel free to reach out at faikimiutai3@gmail.com or open an issue in the repository.

You can modify the sections based on your project’s final structure. Let me know if you need more details!
