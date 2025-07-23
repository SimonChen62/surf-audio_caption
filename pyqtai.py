import sys
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import wave
from io import BytesIO

from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFileDialog,
    QStackedWidget, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap, QDragEnterEvent, QDropEvent


# 尝试导入 inference 模块
try:
    from inference1 import inference
except Exception as e:
    inference = None
    print(f"inference import error: {e}")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

RECORD_DIR = os.path.join(os.path.dirname(__file__), 'recoding')
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)

def get_next_wav_filename():
    files = [f for f in os.listdir(RECORD_DIR) if f.endswith('.wav')]
    nums = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(RECORD_DIR, f"{next_num:03d}.wav")


class LoadingIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("loading")
        self.setFont(QFont("Consolas", 14))
        self.setStyleSheet("color: #ff4d4d;")
        self.dots = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dots)

    def start(self):
        self.dots = 0
        self.timer.start(500)

    def stop(self):
        self.timer.stop()
        self.setText("")

    def update_dots(self):
        self.dots = (self.dots + 1) % 4
        self.setText("loading" + "." * self.dots)


class DragDropWidget(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2d2d2d);
                border-radius: 18px;
                border: 2px dashed #00bfff;
                padding: 20px;
            }
        """)
        self.label = QLabel("🎵 Drag or select a .wav file", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #00bfff;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.clear_timer = QTimer(self)
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self.clear_label)

    def clear_label(self):
        self.label.setText("🎵 Drag or select a .wav file")

    def set_temp_label(self, text):
        self.label.setText(text)
        self.clear_timer.start(3000)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.wav'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.wav'):
                self.save_wav(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("WAV Files (*.wav)")
            if file_dialog.exec():
                file_path = file_dialog.selectedFiles()[0]
                self.save_wav(file_path)

    def save_wav(self, src_path):
        dst_path = get_next_wav_filename()
        with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
            dst.write(src.read())
        self.label.setText(f"Saved: {os.path.basename(dst_path)}")
        self.file_dropped.emit(dst_path)


class RecordWidget(QFrame):
    record_status = pyqtSignal(str)
    record_time = pyqtSignal(str)
    caption_result = pyqtSignal(str)
    realtime_wave = pyqtSignal(np.ndarray)
    ask_analyze = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border-radius: 18px;
                border: 2px solid #00bfff;
                padding: 15px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.record_btn = QPushButton("● Record", self)
        self.record_btn.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00bfff, stop:1 #007acc);
                color: #fff;
                border-radius: 12px;
                padding: 12px 0;
                font-size: 18px;
                min-height: 40px;
            }
            QPushButton:hover {
                background: #00bfff;
                color: #000;
            }
        """)
        self.record_btn.clicked.connect(self.start_record)
        self.layout.addWidget(self.record_btn)
        self.setLayout(self.layout)
        self.is_recording = False
        self.is_paused = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.elapsed_seconds = 0
        self.time_label = QLabel("00:00", self)
        self.time_label.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #00bfff;")
        self.layout.addWidget(self.time_label)
        self.time_label.hide()
        self.audio_thread = None
        self.audio_stream = None
        self.audio_frames = []
        self.wav_path = None
        self.p = None
        self.lock = threading.Lock()
        self.ask_analyze.connect(self.on_ask_analyze)

    def start_record(self):
        if not PYAUDIO_AVAILABLE:
            self.record_status.emit("pyaudio is not installed, cannot record!")
            return
        self.record_btn.hide()
        self.pause_btn = QPushButton("⏸ Pause", self)
        self.pause_btn.setFont(QFont("Segoe UI", 13))
        self.pause_btn.setStyleSheet("""
            background:#333;
            color:#fff;
            border-radius:8px;
            padding: 8px;
            margin: 5px 0;
        """)
        self.stop_btn = QPushButton("■ Stop", self)
        self.stop_btn.setFont(QFont("Segoe UI", 13))
        self.stop_btn.setStyleSheet("""
            background:#d9534f;
            color:#fff;
            border-radius:8px;
            padding: 8px;
            margin: 5px 0;
        """)
        self.pause_btn.clicked.connect(self.pause_record)
        self.stop_btn.clicked.connect(self.stop_record)
        self.layout.addWidget(self.pause_btn)
        self.layout.addWidget(self.stop_btn)
        self.is_recording = True
        self.is_paused = False
        self.elapsed_seconds = 0
        self.time_label.setText("00:00")
        self.time_label.show()
        self.timer.start(1000)
        self.wav_path = get_next_wav_filename()
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()
        self.record_status.emit(f"Start recording... File will be saved to: {self.wav_path}")

    def update_time(self):
        if self.is_recording and not self.is_paused:
            self.elapsed_seconds += 1
            m, s = divmod(self.elapsed_seconds, 60)
            self.time_label.setText(f"{m:02d}:{s:02d}")
            self.record_time.emit(self.time_label.text())

    def record_audio(self):
        self.p = pyaudio.PyAudio()
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        self.audio_stream = stream
        while self.is_recording:
            if self.is_paused:
                import time
                time.sleep(0.1)
                continue
            data = stream.read(1024, exception_on_overflow=False)
            with self.lock:
                self.audio_frames.append(data)
                audio_np = np.frombuffer(b''.join(self.audio_frames), dtype=np.int16)
                self.realtime_wave.emit(audio_np)
        stream.stop_stream()
        stream.close()
        self.p.terminate()
        wf = wave.open(self.wav_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        with self.lock:
            for frame in self.audio_frames:
                wf.writeframes(frame)
        wf.close()
        self.record_status.emit(f"Recording saved: {os.path.basename(self.wav_path)}")
        self.ask_analyze.emit(self.wav_path)

    def on_ask_analyze(self, wav_path):
        reply = QMessageBox.question(self, "Analyze the recording",
                                    "The recording has been saved. Do you want to analyze it immediately?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if inference:
                self.record_status.emit("Analyzing audio, please wait...")
                threading.Thread(target=self.run_inference, args=(wav_path,), daemon=True).start()
            else:
                self.record_status.emit("Inference module not available")

    def run_inference(self, wav_path):
        try:
            result = inference(wav_path)
            self.caption_result.emit(f"AI Caption: {result}")
        except Exception as e:
            self.caption_result.emit(f"AI inference error: {str(e)}")

    def pause_record(self):
        if not self.is_paused:
            self.pause_btn.setText("▶ Resume")
            self.is_paused = True
            self.record_status.emit("Recording paused")
        else:
            self.pause_btn.setText("⏸ Pause")
            self.is_paused = False
            self.record_status.emit("Recording resumed...")

    def stop_record(self):
        self.pause_btn.deleteLater()
        self.stop_btn.deleteLater()
        self.record_btn.show()
        self.is_recording = False
        self.is_paused = False
        self.timer.stop()
        self.time_label.hide()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join()
        self.record_status.emit("Recording stopped")


class WaveformWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setStyleSheet("background: #1a1a1a; border-radius: 14px;")

    def plot_wave(self, wav_path):
        try:
            import librosa
            y, sr = librosa.load(wav_path, sr=16000)
            fig, ax = plt.subplots(figsize=(7, 1.2), dpi=100)
            ax.plot(y, color="#00bfff", linewidth=1.5)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            buf.seek(0)
            pix = QPixmap()
            pix.loadFromData(buf.read())
            self.setPixmap(pix)
        except Exception as e:
            self.setText(f"Waveform display error: {str(e)}")

    def plot_realtime_wave(self, y):
        if len(y) == 0:
            return
        try:
            fig, ax = plt.subplots(figsize=(7, 1.2), dpi=100)
            ax.plot(y, color="#00bfff", linewidth=1.5)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            buf.seek(0)
            pix = QPixmap()
            pix.loadFromData(buf.read())
            self.setPixmap(pix)
        except Exception as e:
            print(f"Realtime wave error: {e}")


class StartPage(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)

        title = QLabel("🎤 Audio Captioning Demo")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00bfff; letter-spacing:2px; margin-top:10px;")
        main_layout.addWidget(title)

        diagram = QLabel(self)
        diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        diagram.setStyleSheet("background: #1a1a1a; border-radius: 14px; margin: 8px;")
        diagram.setFixedHeight(300)
        diagram_path = os.path.join(os.path.dirname(__file__), "captioning_diagram.png")
        if os.path.exists(diagram_path):
            pix = QPixmap(diagram_path)
            diagram.setPixmap(pix.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))
        else:
            diagram.setText("Captioning 模块流程图\n[请放置 captioning_diagram.png]")
            diagram.setStyleSheet("color:#bbb; background: #1a1a1a; border-radius: 14px;")
        main_layout.addWidget(diagram)

        self.waveform = WaveformWidget(self)
        main_layout.addWidget(self.waveform)

        self.output_box = QTextEdit(self)
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("System output...")
        self.output_box.setFont(QFont("Consolas", 14))
        self.output_box.setStyleSheet("""
            QTextEdit {
                background: #181c20;
                color: #d4d4d4;
                border-radius: 12px;
                border: 1px solid #00bfff;
                padding: 10px;
                margin: 8px 0;
            }
        """)
        main_layout.addWidget(self.output_box, 2)

        bottom_layout = QHBoxLayout()
        self.record_widget = RecordWidget(self)
        self.record_widget.setFixedWidth(160)
        self.dragdrop_widget = DragDropWidget(self)
        bottom_layout.addWidget(self.record_widget, 1)
        bottom_layout.addWidget(self.dragdrop_widget, 3)
        main_layout.addLayout(bottom_layout, 1)

        copyright = QLabel("© 2025 Audio Captioning Demo | Powered by PyQt6")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright.setStyleSheet("color:#888; font-size:12px; margin-top:10px;")
        main_layout.addWidget(copyright)
        self.setLayout(main_layout)

        self.loading_indicator = LoadingIndicator(self)
        self.loading_indicator.move(20, self.height() - 40)
        self.loading_indicator.hide()

        # ToolTips
        self.record_widget.setToolTip("Click to start recording")
        self.dragdrop_widget.setToolTip("Drag and drop WAV files here")
        self.output_box.setToolTip("AI generated results will be displayed here")

        # Connect signals
        self.record_widget.realtime_wave.connect(self.waveform.plot_realtime_wave)
        self.record_widget.record_status.connect(self.append_output)
        self.record_widget.record_time.connect(lambda t: None)
        self.record_widget.caption_result.connect(self.append_ai_output)
        self.dragdrop_widget.file_dropped.connect(self.handle_file_dropped)

    def resizeEvent(self, event):
        self.loading_indicator.move(20, self.height() - 40)
        super().resizeEvent(event)

    def append_output(self, msg):
        html = f'<span style="color:#4ec9b0;">{msg}</span>'
        self.output_box.append(html)

    def append_ai_output(self, msg):
        html = f'<span style="color:#569cd6;">{msg}</span>'
        self.output_box.append(html)

    def handle_file_dropped(self, dst_path):
        self.append_output(f"File saved to: {dst_path}")
        self.dragdrop_widget.set_temp_label(f"Saved: {os.path.basename(dst_path)}")
        self.waveform.plot_wave(dst_path)

        filename = os.path.basename(dst_path)
        self.output_box.append(f"<b>Loaded:</b> {filename}")

        if inference:
            self.loading_indicator.show()
            self.loading_indicator.start()

            def run_infer():
                try:
                    result = inference(dst_path)
                    self.append_ai_output(f"AI Caption: {result}")
                except Exception as e:
                    self.append_ai_output(f"AI inference error: {str(e)}")
                finally:
                    QTimer.singleShot(0, self.loading_indicator.hide)
                    QTimer.singleShot(0, self.loading_indicator.stop)

            threading.Thread(target=run_infer, daemon=True).start()
        else:
            self.append_output("Inference module not available")


class AboutPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("About This Program")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        label.setStyleSheet("color: #00bfff;")
        layout.addWidget(label)

        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setFont(QFont("Segoe UI", 14))
        desc.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                color: #ffffff;
                border-radius: 10px;
                border: 1px solid #00bfff;
                padding: 15px;
            }
        """)
        desc_text = """
        <h3>Welcome to Audio Captioning Demo</h3>
        <p>This application demonstrates how AI can automatically generate captions for audio files using deep learning models.</p>
        <p><strong>Functionality:</strong></p>
        <ul>
            <li>Record audio via microphone</li>
            <li>Load existing WAV files</li>
            <li>Use AI to generate descriptive captions</li>
            <li>Display waveform of loaded audio</li>
        </ul>
        """
        desc.setHtml(desc_text)
        layout.addWidget(desc)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioCaption")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "icon.png")))
        self.resize(980, 680)

        main_widget = QWidget(self)
        main_layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        title_label = QLabel("Audio\nCaption")
        title_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #00bfff; letter-spacing:2px;")
        left_layout.addWidget(title_label)

        btn1_frame = QFrame()
        btn1_layout = QVBoxLayout(btn1_frame)
        self.start_btn = QPushButton("Start")
        self.start_btn.setFont(QFont("Segoe UI", 16))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #252526;
                color: #d4d4d4;
                border-radius: 10px;
                padding: 12px 0;
                margin: 5px 0;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00bfff, stop:1 #007acc);
                color: #fff;
                border: none;
            }
        """)
        btn1_layout.addWidget(self.start_btn)
        btn1_frame.setLayout(btn1_layout)
        left_layout.addWidget(btn1_frame)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #00bfff;")
        left_layout.addWidget(line)

        btn2_frame = QFrame()
        btn2_layout = QVBoxLayout(btn2_frame)
        self.about_btn = QPushButton("About")
        self.about_btn.setFont(QFont("Segoe UI", 16))
        self.about_btn.setStyleSheet("""
            QPushButton {
                background: #252526;
                color: #d4d4d4;
                border-radius: 10px;
                padding: 12px 0;
                margin: 5px 0;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00bfff, stop:1 #007acc);
                color: #fff;
                border: none;
            }
        """)
        btn2_layout.addWidget(self.about_btn)
        btn2_frame.setLayout(btn2_layout)
        left_layout.addWidget(btn2_frame)

        left_layout.addStretch(1)
        left_frame = QFrame()
        left_frame.setLayout(left_layout)
        left_frame.setFixedWidth(220)
        left_frame.setStyleSheet("background: #1e1e1e; border-right: 2px solid #00bfff;")

        self.stack = QStackedWidget()
        self.start_page = StartPage()
        self.about_page = AboutPage()
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.about_page)

        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(self.stack, 4)
        self.setCentralWidget(main_widget)

        self.setStyleSheet("""
            QMainWindow { background: #000000; }
            QWidget { background: #000000; color: #d4d4d4; }
            QToolTip {
                background-color: #00bfff;
                color: #000000;
                border: 1px solid #007acc;
                padding: 5px;
                border-radius: 5px;
            }
        """)

        self.start_btn.clicked.connect(self.show_start)
        self.about_btn.clicked.connect(self.show_about)
        self.show_start()

    def show_start(self):
        self.stack.setCurrentWidget(self.start_page)

    def show_about(self):
        self.stack.setCurrentWidget(self.about_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())