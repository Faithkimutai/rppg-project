from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel,
    QPushButton, QCheckBox, QFileDialog, QHBoxLayout, QSizePolicy, QTextEdit, QGridLayout
)
from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import QtGui
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import time
from PyQt6.QtCore import (
    Qt, 
    QPropertyAnimation, 
    QRect, 
    QEasingCurve
)
from PyQt6.QtGui import QFont
import random

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

basedir = os.path.dirname(__file__)

try:
    from ctypes import windll  
    myappid = 'com.bpsoftware.001'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("BP Prediction App")
        self.setGeometry(200, 200, 800, 600)
        
        # Create stacked widget as central widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Initialize all pages
        self.intro_page = IntroPage(self)
        self.terms_page = TermsPage(self)
        self.image_page = ImagePage(self)
        self.signal_page = SignalPage(self)
        self.prediction_page = PredictionPage(self)
        
        # Add all pages to stack
        self.stack.addWidget(self.intro_page)
        self.stack.addWidget(self.terms_page)
        self.stack.addWidget(self.image_page)
        self.stack.addWidget(self.signal_page)
        self.stack.addWidget(self.prediction_page)
        
        # Connect navigation signals
        self.connect_signals()
        
        # Start with intro page
        self.stack.setCurrentWidget(self.intro_page)

    def connect_signals(self):
        """Connect navigation signals using flexible approach"""
        # Intro page (using getattr for safety)
        if hasattr(self.intro_page, 'start_button'):
            self.intro_page.start_button.clicked.connect(
                lambda: self.stack.setCurrentWidget(self.terms_page))
        
        # Terms page
        if hasattr(self.terms_page, 'next_button'):
            self.terms_page.next_button.clicked.connect(
                lambda: self.stack.setCurrentWidget(self.image_page))
        
        # Image page
        if hasattr(self.image_page, 'next_button'):
            self.image_page.next_button.clicked.connect(
                lambda: self.stack.setCurrentWidget(self.signal_page))
        
        # Signal page
        if hasattr(self.signal_page, 'next_button'):
            self.signal_page.next_button.clicked.connect(
                self.show_prediction)

    def show_prediction(self):
        """Handle transition to prediction page with animation"""
        self.prediction_page.transition_from_signal(self.signal_page)
        self.stack.setCurrentWidget(self.prediction_page)

    def closeEvent(self, event):
        """Clean up resources when closing"""
        if hasattr(self.signal_page, 'cap'):
            self.signal_page.cap.release()
        event.accept()
        
# Intro Page
class IntroPage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Welcome to the Blood Pressure Prediction AI App")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        ai_label = QLabel("AI Powered")
        ai_label.setFont(QFont("Arial", 10))
        ai_label.setStyleSheet("color: purple;")

        start_button = QPushButton("Get Started")
        start_button.setStyleSheet("padding: 10px; font-size: 18px; background-color: #4CAF50; color: white; border-radius: 10px;")
        start_button.clicked.connect(lambda: main_app.stack.setCurrentWidget(main_app.terms_page))

        layout.addWidget(title)
        layout.addWidget(ai_label)
        layout.addWidget(start_button)
        self.setLayout(layout)

# Terms Page
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, 
    QPushButton, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class TermsPage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        
        # Main layout with proper spacing
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        
        # Scroll content widget
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: white;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(40, 20, 40, 20)
        scroll_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Terms of Service")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #f542e6; margin-bottom: 15px;")
        
        # Terms content with scrollable text
        terms_content = QTextEdit()
        terms_content.setReadOnly(True)
        terms_content.setFrameShape(QFrame.Shape.NoFrame)
        terms_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        terms_content.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                color: #333333;
                line-height: 1.5;
                padding: 10px;
                border: none;
                background: transparent;
            }
        """)

        # HTML formatted terms text
        terms_text = """
        <h2 style="color: #f542e6; text-align: center;">Blood Pressure Prediction Application Terms of Use</h2>
        
        <h3 style="color: #f542e6;">1. Medical Disclaimer</h3>
        <p>The blood pressure predictions provided by this application are for informational purposes only 
        and should not be considered as medical advice. This application is not a substitute for professional 
        medical diagnosis, treatment, or advice. Always consult with a qualified healthcare provider regarding 
        any medical concerns.</p>
        
        <h3 style="color: #f542e6;">2. Data Accuracy</h3>
        <p>While we strive to provide accurate predictions, we cannot guarantee the precision of the results. 
        Blood pressure can be affected by numerous factors that this application may not account for. 
        Actual measurements may vary from predicted values.</p>
        
        <h3 style="color: #f542e6;">3. Liability Limitation</h3>
        <p>The developers of this application shall not be held liable for any damages, health complications, 
        or adverse outcomes resulting from the use of or reliance on the information provided by this application. 
        Users assume all risks associated with using this application.</p>
        
        <h3 style="color: #f542e6;">4. Data Privacy</h3>
        <p>Any health data you input will be processed locally on your device and will not be transmitted to 
        external servers without your explicit consent. We respect your privacy and will handle your health 
        information in accordance with applicable data protection laws.</p>
        
        <h3 style="color: #f542e6;">5. Appropriate Use</h3>
        <p>This application is intended for use by adults only. It should not be used to diagnose, treat, cure, 
        or prevent any disease without supervision from a licensed medical professional.</p>
        
        <h3 style="color: #f542e6;">6. Updates and Changes</h3>
        <p>We reserve the right to modify these terms at any time. Continued use of the application after such 
        modifications constitutes your acceptance of the new terms.</p>
        
        <h3 style="color: #f542e6;">7. User Responsibilities</h3>
        <p>You agree to use this application responsibly and to consult with a healthcare professional before 
        making any medical decisions based on the application's outputs.</p>
        """
        terms_content.setHtml(terms_text)
        
        # Acceptance section (fixed at bottom)
        accept_frame = QFrame()
        accept_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        accept_frame.setFrameShape(QFrame.Shape.StyledPanel)
        accept_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        accept_layout = QVBoxLayout(accept_frame)
        accept_layout.setContentsMargins(10, 10, 10, 10)
        accept_layout.setSpacing(15)
        
        # Checkbox
        terms_checkbox = QCheckBox("I have read and accept the terms and conditions")
        terms_checkbox.setFont(QFont("Arial", 12))
        terms_checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 10px;
                color: #333333;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        
        # Continue button
        next_button = QPushButton("Continue to Image Capture")
        next_button.setEnabled(False)
        next_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        next_button.setMinimumHeight(45)
        next_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QPushButton:hover:enabled {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
        """)
        next_button.clicked.connect(lambda: main_app.stack.setCurrentWidget(main_app.image_page))
        
        # Connect checkbox to button state
        terms_checkbox.stateChanged.connect(next_button.setEnabled)
        
        # Add widgets to acceptance frame
        accept_layout.addWidget(terms_checkbox, 0, Qt.AlignmentFlag.AlignLeft)
        accept_layout.addWidget(next_button, 0, Qt.AlignmentFlag.AlignRight)
        
        # Add widgets to scroll layout
        scroll_layout.addWidget(title_label)
        scroll_layout.addWidget(terms_content, 1)  # Takes remaining space
        scroll_layout.addWidget(accept_frame)
        
        # Set scroll content
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Ensure proper sizing
        self.setMinimumSize(800, 600)
        scroll_content.setMinimumSize(600, 400)

# Image Capture Page
class ImagePage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.image_path = None
        self.frame_buffer = []
        
        # Main layout with proper spacing and margins
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)
        
        # Title section
        title_label = QLabel("Upload Facial Image or capture")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        
        # Preview pane with improved styling
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        
        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 14px;
            }
        """)
        self.preview_label.setFixedSize(500, 350)
        preview_layout.addWidget(self.preview_label)
        
        # Button container
        button_container = QHBoxLayout()
        button_container.setSpacing(15)
        
        # Upload button with icon
        upload_button = QPushButton(" Upload Image")
        upload_button.setIcon(QIcon.fromTheme("document-open"))
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                min-width: 150px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        upload_button.clicked.connect(self.upload_image)
        
        # Webcam button with icon
        webcam_button = QPushButton(" Open Webcam")
        webcam_button.setIcon(QIcon.fromTheme("camera-web"))
        webcam_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                min-width: 150px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        webcam_button.clicked.connect(self.start_webcam)
        
        button_container.addWidget(upload_button)
        button_container.addWidget(webcam_button)
        
        # Next button with improved styling
        self.next_button = QPushButton("Proceed")
        self.next_button.setEnabled(False)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                min-width: 200px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton:hover:enabled {
                background-color: #8e44ad;
            }
        """)
        self.next_button.clicked.connect(self.process_rppg)
        
        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(preview_frame)
        main_layout.addLayout(button_container)
        main_layout.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.setLayout(main_layout)
        
        # Timer for webcam capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Facial Image", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.update_preview()
            self.next_button.setEnabled(True)
            self.preview_label.setStyleSheet("")  # Remove placeholder styling

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.preview_label.setText("Looking for camera...")
            self.timer.start(100)
        else:
            self.preview_label.setText("Could not access camera")
            self.preview_label.setStyleSheet("color: #e74c3c;")

    def capture_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    self.image_path = temp_file.name
                    cv2.imwrite(self.image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                self.update_preview()
                self.cap.release()
                self.timer.stop()
                self.next_button.setEnabled(True)

    def update_preview(self):
        if self.image_path and os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.width(), 
                    self.preview_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.preview_label.setText("")

    def process_rppg(self):
        # Simulate face region data (replace with actual processing)
        face_region = np.random.rand(30)  
        filtered_signal = filtfilt(*butter(3, 0.1, btype='low'), face_region)
        heart_rate = int(len(find_peaks(filtered_signal, distance=15)[0]) * 2)

        # Create and save signal plot
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            signal_image_path = temp_file.name
            plt.figure(figsize=(5, 2))
            plt.plot(filtered_signal, color='red', linewidth=2)
            plt.axis("off")
            plt.savefig(signal_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

        # Pass results to signal page
        self.main_app.signal_page.display_results(
            self.image_path, 
            filtered_signal, 
            heart_rate
        )
        self.main_app.stack.setCurrentWidget(self.main_app.signal_page)

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QScrollArea, QGridLayout,
    QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SignalPage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.signal_data = np.array([])
        self.live_signal = []
        self.cap = None
        self.bp_values = []

        # Main scroll area setup
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Main content widget
        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setContentsMargins(20, 10, 20, 20)
        self.layout.setSpacing(15)

        # Title
        title = QLabel("Your Statistics")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        self.layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Image and signal plots
        plots_row = QHBoxLayout()
        plots_row.setSpacing(15)

        # Facial image preview
        img_frame = QFrame()
        img_frame.setFixedSize(300, 180)
        img_frame.setStyleSheet("background: #f8f9fa; border-radius: 8px;")
        img_layout = QVBoxLayout(img_frame)
        self.image_label = QLabel("Facial Image Preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #7f8c8d;")
        img_layout.addWidget(self.image_label)
        plots_row.addWidget(img_frame)

        # Static rPPG plot
        static_plot_frame = QFrame()
        static_plot_frame.setFixedSize(300, 180)
        static_plot_frame.setStyleSheet("background: white; border-radius: 8px;")
        static_layout = QVBoxLayout(static_plot_frame)
        self.fig_static = plt.figure(figsize=(4, 2.5), tight_layout=True)
        self.ax_static = self.fig_static.add_subplot(111)
        self.ax_static.set_title("Processed rPPG", fontsize=9)
        self.ax_static.tick_params(labelsize=7)
        self.canvas_static = FigureCanvas(self.fig_static)
        static_layout.addWidget(self.canvas_static)
        plots_row.addWidget(static_plot_frame)

        self.layout.addLayout(plots_row)

        # Live signal plot
        live_frame = QFrame()
        live_frame.setMinimumHeight(150)
        live_frame.setStyleSheet("background: white; border-radius: 8px;")
        live_layout = QVBoxLayout(live_frame)
        self.fig_live = plt.figure(figsize=(8, 1.8), tight_layout=True)
        self.ax_live = self.fig_live.add_subplot(111)
        self.ax_live.set_title("Live Signal", fontsize=9)
        self.ax_live.tick_params(labelsize=7)
        self.signal_line, = self.ax_live.plot([], [], color="#2ecc71", linewidth=1)
        self.canvas_live = FigureCanvas(self.fig_live)
        live_layout.addWidget(self.canvas_live)
        self.layout.addWidget(live_frame)

        # Vital signs display
        vitals_frame = QFrame()
        vitals_frame.setStyleSheet("background: #ecf0f1; border-radius: 8px;")
        vitals_layout = QGridLayout(vitals_frame)
        vitals_layout.setContentsMargins(15, 10, 15, 10)
        vitals_layout.setVerticalSpacing(5)
        vitals_layout.setHorizontalSpacing(20)

        # Heart Rate
        self.heart_rate = self.create_vital_sign(vitals_layout, "HEART RATE", "#e74c3c", "BPM", 0)

        # SpO2
        self.spo2 = self.create_vital_sign(vitals_layout, "OXYGEN SATURATION", "#3498db", "%", 1)

        # HRV
        self.hrv = self.create_vital_sign(vitals_layout, "HR VARIABILITY", "#2ecc71", "ms", 2)

        # BP Prediction
        self.bp_pred = self.create_vital_sign(vitals_layout, "BP PREDICTION", "#9b59b6", "mmHg", 3)

        self.layout.addWidget(vitals_frame)

        # Set up scroll area
        scroll.setWidget(content)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)
        self.vitals_timer = QTimer()
        self.vitals_timer.timeout.connect(self.update_vitals)

    def create_vital_sign(self, layout, title, color, unit, column):
        """Helper function to create a vital sign display."""
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #7f8c8d;")
        value_label = QLabel("--")
        value_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color};")
        unit_label = QLabel(unit)
        unit_label.setFont(QFont("Arial", 9))
        layout.addWidget(title_label, 0, column, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label, 1, column, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(unit_label, 2, column, Qt.AlignmentFlag.AlignCenter)
        return value_label

    def start_live_animation(self):
        """Initializes real-time webcam feed."""
        self.live_signal = []
        self.bp_values = []
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(100)
            self.vitals_timer.start(5000)
        else:
            self.heart_rate.setText("--")
            self.heart_rate.setStyleSheet("color: #e74c3c;")
    
    def display_results(self, image_path, signal_data, red_channel):
        """Display the analysis results from the captured image"""
        # Load and display image
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.image_label.setText("Image Not Available")
            self.image_label.setStyleSheet("color: #e74c3c;")

        # Calculate vital signs
        heart_rate = self._calculate_heart_rate(signal_data)
        spo2 = self._calculate_spo2(red_channel, signal_data)
        hrv = self._calculate_hrv(signal_data)
        
        # Update UI
        self.heart_rate.setText(f"{heart_rate}")
        self.spo2.setText(f"{spo2}")
        self.hrv.setText(f"{hrv}")
        self.bp_pred.setText(f"{np.random.randint(112, 127)}/{np.random.randint(72, 86)}")

        # Plot signal data
        self._plot_signal(signal_data)
    
        self.start_live_animation()

    def _calculate_heart_rate(self, signal_data):
        """Calculate heart rate from signal data"""
        try:
            peaks, _ = find_peaks(signal_data, distance=15)
            return int(len(peaks) * 2)  # Convert peaks to BPM
        except:
            return "--"

    def _calculate_spo2(self, red_signal, green_signal):
        """Estimate oxygen saturation"""
        try:
            ratio = np.mean(red_signal) / np.mean(green_signal)
            spo2 = 100 - (5 * (ratio - 1))
            return round(max(90, min(100, spo2)))
        except:
            return "--"

    def _calculate_hrv(self, signal_data):
        """Calculate heart rate variability"""
        try:
            peaks, _ = find_peaks(signal_data, distance=30)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) * (1000 / 30)  # Convert to ms
                return round(np.std(rr_intervals))
            return "--"
        except:
            return "--"

    def _plot_signal(self, signal_data):
        """Plot the processed signal"""
        self.ax_static.clear()
        self.ax_static.plot(signal_data, color="#e74c3c", linewidth=1.5)
        self.ax_static.set_title("Processed rPPG Signal", fontsize=9)
        self.ax_static.set_xlabel("Time (frames)", fontsize=8)
        self.ax_static.set_ylabel("Amplitude", fontsize=8)
        self.canvas_static.draw()

    def update_live_feed(self):
        """Captures webcam frames and extracts real-time rPPG signal."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                green_channel = np.mean(frame[:, :, 1])
                self.live_signal.append(green_channel)

                if len(self.live_signal) > 100:
                    self.live_signal.pop(0)

                self.update_live_plot()

    def update_live_plot(self):
        """Updates the live rPPG animation."""
        if len(self.live_signal) < 10:
            return

        self.ax_live.clear()
        self.ax_live.set_title("Live rPPG Signal", fontsize=10)
        self.ax_live.plot(self.live_signal, color="#2ecc71", linewidth=1.5)
        self.ax_live.set_xlabel("Time (frames)", fontsize=8)
        self.ax_live.set_ylabel("Amplitude", fontsize=8)
        self.canvas_live.draw()

    def update_vitals(self):
        """Simulates vital sign updates."""
        if len(self.live_signal) < 30:
            return

        self.heart_rate.setText(f"{np.random.randint(60, 100)}")
        self.spo2.setText(f"{np.random.randint(95, 100)}")
        self.hrv.setText(f"{np.random.randint(30, 50)}")

    def closeEvent(self, event):
        """Releases webcam on window close."""
        if self.cap:
            self.cap.release()
        event.accept()


class PredictionPage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.signal_widget = None
        self.animation = None
        
        # Main layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Left container for signal page content
        self.left_container = QWidget()
        self.left_container.setStyleSheet("background: white;")
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Right container for prediction
        self.right_container = QWidget()
        self.right_container.setStyleSheet("background: #f8f9fa;")
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(40, 40, 40, 40)
        self.right_layout.setSpacing(20)
        
        # Add containers to main layout
        self.main_layout.addWidget(self.left_container, 60)  # 60% width
        self.main_layout.addWidget(self.right_container, 40)  # 40% width
        
        # Initialize prediction UI
        self.init_prediction_ui()
        
    def init_prediction_ui(self):
        """Initialize the prediction display area"""
        # Title
        title = QLabel("Blood Pressure Prediction")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # BP value display
        bp_frame = QFrame()
        bp_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
            }
        """)
        bp_layout = QHBoxLayout(bp_frame)
        bp_layout.setContentsMargins(20, 20, 20, 20)
        
        # Systolic
        systolic_widget = self.create_bp_widget("SYSTOLIC", "112-127", "mmHg", "#e74c3c")
        # Diastolic
        diastolic_widget = self.create_bp_widget("DIASTOLIC", "72-86", "mmHg", "#3498db")
        
        bp_layout.addWidget(systolic_widget)
        bp_layout.addWidget(diastolic_widget)
        
        # Analysis section
        analysis_title = QLabel("Vital Signs Analysis")
        analysis_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        analysis_title.setStyleSheet("color: #2c3e50; margin-top: 20px;")
        
        # Vital signs display
        vitals_frame = QFrame()
        vitals_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                padding: 15px;
            }
        """)
        vitals_layout = QVBoxLayout(vitals_frame)
        
        self.heart_rate_label = self.create_vital_label("HEART RATE", "--", "BPM", "#e74c3c")
        self.spo2_label = self.create_vital_label("OXYGEN SATURATION", "--", "%", "#3498db")
        self.hrv_label = self.create_vital_label("HR VARIABILITY", "--", "ms", "#2ecc71")
        
        vitals_layout.addWidget(self.heart_rate_label)
        vitals_layout.addWidget(self.spo2_label)
        vitals_layout.addWidget(self.hrv_label)
        
        # Add widgets to right layout
        self.right_layout.addWidget(title)
        self.right_layout.addWidget(bp_frame)
        self.right_layout.addWidget(analysis_title)
        self.right_layout.addWidget(vitals_frame)
        self.right_layout.addStretch()
        
    def create_bp_widget(self, title, value, unit, color):
        """Create a BP value display widget"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {color};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color};")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        unit_label = QLabel(unit)
        unit_label.setFont(QFont("Arial", 10))
        unit_label.setStyleSheet(f"color: {color};")
        unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(unit_label)
        
        return widget
        
    def create_vital_label(self, title, value, unit, color):
        """Create a vital sign display widget"""
        widget = QFrame()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10))
        title_label.setStyleSheet(f"color: #7f8c8d; min-width: 120px;")
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color}; min-width: 50px;")
        
        unit_label = QLabel(unit)
        unit_label.setFont(QFont("Arial", 10))
        unit_label.setStyleSheet("color: #7f8c8d;")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(unit_label)
        layout.addStretch()
        
        return widget
        
    def transition_from_signal(self, signal_page):
        """Animate transition from signal page"""
        self.signal_widget = signal_page
        
        # Clone the signal page content
        signal_clone = QWidget()
        signal_clone.setLayout(signal_page.layout())
        
        # Add to left container
        self.left_layout.addWidget(signal_clone)
        
        # Set initial state for animation
        self.left_container.setGeometry(0, 0, self.width(), self.height())
        
        # Create animation
        self.animation = QPropertyAnimation(self.left_container, b"geometry")
        self.animation.setDuration(500)
        self.animation.setStartValue(QRect(0, 0, self.width(), self.height()))
        self.animation.setEndValue(QRect(0, 0, int(self.width() * 0.6), self.height()))
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()
        
        # Update prediction with data from signal page
        self.update_prediction()
        
    def update_prediction(self):
        """Update prediction with data from signal page"""
        if not self.signal_widget:
            return
            
        # Generate random BP values in specified ranges
        systolic = random.randint(112, 127)
        diastolic = random.randint(72, 86)
        
        # Update BP display
        bp_frame = self.right_container.findChild(QFrame)
        if bp_frame:
            systolic_label = bp_frame.findChildren(QLabel)[1]  # First value label
            diastolic_label = bp_frame.findChildren(QLabel)[4]  # Second value label
            systolic_label.setText(str(systolic))
            diastolic_label.setText(str(diastolic))
        
        # Update vital signs from signal page
        self.heart_rate_label.findChild(QLabel, "", 1).setText(self.signal_widget.heart_rate_label.text())
        self.spo2_label.findChild(QLabel, "", 1).setText(self.signal_widget.spo2_label.text())
        self.hrv_label.findChild(QLabel, "", 1).setText(self.signal_widget.hrv_label.text())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('assets/cardiogram.ico'))
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec())
