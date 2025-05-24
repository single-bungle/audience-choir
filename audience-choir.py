import sys
import os
import tempfile
import soundfile as sf
import numpy as np
import time
from datetime import datetime
from argparse import Namespace
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QLabel, QAbstractItemView,
    QFormLayout, QLineEdit, QSpinBox, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QCheckBox, 
    QListWidget, QListWidgetItem, QSlider, QScrollArea
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from pydub import AudioSegment
import librosa
from scipy.signal import fftconvolve
import pyloudnorm as pyln

# 모델 코드가 있는 디렉토리를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'seed_vc'))

# BRIR 디렉토리 설정
BRIR_DIR = os.path.join(os.path.dirname(__file__), 'D1-Brir')

def convolve_hrir(signal, hrir_L, hrir_R):
    left_output = fftconvolve(signal, hrir_L, mode='full')
    right_output = fftconvolve(signal, hrir_R, mode='full')
    output = np.vstack((left_output, right_output)).T
    return output

def spherical_to_cartesian(azimuth, elevation):
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    return np.array([x, y, z])

def slerp(v0, v1, t):
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return (1 - t) * v0 + t * v1
    return (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1

def interpolated_brir(brir_type, input_location):
    if brir_type == 'D1_brir':
        coordinates = [
            [198.4, -17.5], [251.6, -17.5], [225.0, 64.8], [0.0, 90.0], [341.6, 17.5], 
            [288.4, -17.5], [45.0, 35.3], [315.0, -64.8], [270.0, 45.0], [251.6, 17.5], 
            [288.4, 17.5], [71.6, -17.5], [90.0, 45.0], [90.0, -45.0], [161.6, 17.5], 
            [0.0, -45.0], [315.0, -35.3], [45.0, 64.8], [341.6, -17.5], [0.0, 0.0], 
            [90.0, 0.0], [270.0, 0.0], [225.0, 35.3], [135.0, 0.0], [18.4, -17.5], 
            [18.4, 17.5], [135.0, -35.3], [108.4, -17.5], [198.4, 17.5], [315.0, 35.3], 
            [45.0, -64.8], [0.0, -90.0], [225.0, -35.3], [180.0, -45.0], [135.0, 64.8], 
            [161.6, -17.5], [135.0, 35.3], [315.0, 0.0], [108.4, 17.5], [225.0, -64.8], 
            [180.0, 0.0], [45.0, -35.3], [45.0, 0.0], [0.0, 45.0], [225.0, 0.0], 
            [315.0, 64.8], [71.6, 17.5], [270.0, -45.0], [180.0, 45.0], [135.0, -64.8]
        ]

    cartesian_coords = [spherical_to_cartesian(az, el) for az, el in coordinates]
    desired_azimuth = input_location[0]
    desired_elevation = input_location[1]
    desired_cartesian = spherical_to_cartesian(desired_azimuth, desired_elevation)

    distances = [np.linalg.norm(desired_cartesian - c) for c in cartesian_coords]
    closest_indices = np.argsort(distances)[:2]
    hrir_1_loc = coordinates[closest_indices[0]]
    hrir_2_loc = coordinates[closest_indices[1]]

    hrir_1_path = os.path.join(BRIR_DIR, f"azi_{hrir_1_loc[0]}_ele_{hrir_1_loc[1]}.wav")
    hrir_2_path = os.path.join(BRIR_DIR, f"azi_{hrir_2_loc[0]}_ele_{hrir_2_loc[1]}.wav")

    try:
        hrir_1, sr_hrir_1 = librosa.load(hrir_1_path, sr=None, mono=False)
        hrir_2, sr_hrir_2 = librosa.load(hrir_2_path, sr=None, mono=False)
    except Exception as e:
        raise FileNotFoundError(f"BRIR 파일 로드 실패: {hrir_1_path} 또는 {hrir_2_path} ({str(e)})")

    if sr_hrir_1 != sr_hrir_2:
        raise ValueError("BRIR 파일의 샘플레이트가 일치하지 않습니다.")

    v0, v1 = cartesian_coords[closest_indices[0]], cartesian_coords[closest_indices[1]]
    hrir_1_L, hrir_1_R = hrir_1[0], hrir_1[1]
    hrir_2_L, hrir_2_R = hrir_2[0], hrir_2[1]

    t = distances[closest_indices[0]] / (distances[closest_indices[0]] + distances[closest_indices[1]])
    interpolated_hrir_L = (1 - t) * hrir_1_L + t * hrir_2_L
    interpolated_hrir_R = (1 - t) * hrir_1_R + t * hrir_2_R
    
    return interpolated_hrir_L, interpolated_hrir_R, sr_hrir_1

class TargetAudioControl(QWidget):
    def __init__(self, title, visualizer):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_files = []
        self.audio_data_dict = {}
        self.visualizer = visualizer

        self.upload_button = QPushButton("Upload WAV Files")
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.list_widget = QListWidget()
        self.label = QLabel("No files uploaded.")

        self.upload_button.clicked.connect(self.upload_audio)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)
        self.list_widget.itemChanged.connect(self.toggle_circle_from_item)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.upload_button)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_selected_path = None

    def upload_audio(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select WAV Files", "", "WAV Files (*.wav)")
        if not files:
            return

        for file_path in files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            if file_path not in self.audio_files:
                self.audio_files.append(file_path)

                item = QListWidgetItem(filename)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.list_widget.addItem(item)

                try:
                    audio_data = AudioSegment.from_wav(file_path)
                    self.audio_data_dict[name_without_ext] = audio_data
                except Exception as e:
                    self.label.setText(f"Error loading {filename}: {str(e)}")

        self.label.setText(f"{len(self.audio_files)} files uploaded.")

    def toggle_circle_from_item(self, item):
        filename = item.text()
        name_without_ext = os.path.splitext(filename)[0]

        if item.checkState() == Qt.Checked:
            self.visualizer.add_circle(name_without_ext)
        else:
            if name_without_ext in self.visualizer.circles:
                del self.visualizer.circles[name_without_ext]
                self.visualizer.update()

    def play_audio(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        selected_name = selected_items[0].text()
        for path in self.audio_files:
            if os.path.basename(path) == selected_name:
                self.current_selected_path = path
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
                break

        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play")

class SourceAudioControl(QWidget):
    def __init__(self, title):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_path = None
        self.audio_data = None

        self.upload_button = QPushButton("Upload WAV File")
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.label = QLabel("No file uploaded.")

        self.upload_button.clicked.connect(self.upload_audio)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.upload_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def upload_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a WAV File", "", "WAV Files (*.wav)")
        if not file_path:
            return

        try:
            self.audio_path = file_path
            self.audio_data = AudioSegment.from_wav(file_path)
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.label.setText(f"Error loading file: {str(e)}")

    def play_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play")

    @property
    def current_selected_path(self):
        return self.audio_path

class DraggableTargetCircle(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        self.setStyleSheet("background-color: white;")
        self.circles = {}
        self.dragging_circle = None
        self.drag_offset = QPoint(0, 0)

        self.logic_x_range = (-180, 180)
        self.logic_y_range = (-90, 90)

    def add_circle(self, name):
        center = QPoint(self.width() // 2, self.height() // 2)
        self.circles[name] = center
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        origin_x = w // 2
        origin_y = h // 2

        scale_x = w / (self.logic_x_range[1] - self.logic_x_range[0])
        scale_y = h / (self.logic_y_range[1] - self.logic_y_range[0])

        grid_spacing_deg = 30
        painter.setPen(QPen(QColor(230, 230, 230), 1))

        for lon in range(self.logic_x_range[0], self.logic_x_range[1] + 1, grid_spacing_deg):
            x = origin_x + lon * scale_x
            painter.drawLine(int(x), 0, int(x), h)
            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.darkGray)
            painter.drawText(int(x) , h - 15, 20, 20, Qt.AlignCenter, str(lon))

        for lat in range(self.logic_y_range[0], self.logic_y_range[1] , grid_spacing_deg):
            y = origin_y - lat * scale_y
            painter.drawLine(0, int(y), w, int(y))
            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.darkGray)
            painter.drawText(5, int(y) - 10, 30, 20, Qt.AlignLeft, str(lat))

        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(origin_x, 0, origin_x, h)
        painter.drawLine(0, origin_y, w, origin_y)

        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.setPen(Qt.black)
        painter.drawText(w - 55, h - 115, 60, 20, Qt.AlignCenter, "Azimuth")
        painter.rotate(-90)
        painter.drawText(-50, origin_x + 5, 80, 20, Qt.AlignLeft, "Elevation")
        painter.rotate(90)

        for name, pos in self.circles.items():
            painter.setBrush(QColor(200, 200, 255))
            painter.setPen(QPen(Qt.blue, 2))
            painter.drawEllipse(pos, 15, 15)

            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.black)
            painter.drawText(painter.boundingRect(pos.x() - 30, pos.y() - 10, 60, 20, Qt.AlignCenter, name),
                             Qt.AlignCenter, name)

            logic_x = round((pos.x() - origin_x) / scale_x)
            logic_y = round((origin_y - pos.y()) / scale_y)
            coord_text = f"({logic_x}, {logic_y})"

            painter.setFont(QFont("Arial", 6))
            painter.setPen(Qt.darkGray)
            painter.drawText(pos.x() + 35, pos.y() + 5, coord_text)

    def mousePressEvent(self, event):
        for name, pos in self.circles.items():
            if (pos - event.pos()).manhattanLength() < 30:
                self.dragging_circle = name
                self.drag_offset = event.pos() - pos
                break

    def mouseMoveEvent(self, event):
        if self.dragging_circle:
            new_pos = event.pos() - self.drag_offset

            w = self.width()
            h = self.height()
            origin_x = w // 2
            origin_y = h // 2
            scale_x = w / (self.logic_x_range[1] - self.logic_x_range[0])
            scale_y = h / (self.logic_y_range[1] - self.logic_y_range[0])

            logic_x = (new_pos.x() - origin_x) / scale_x
            logic_y = (origin_y - new_pos.y()) / scale_y

            logic_x = max(self.logic_x_range[0], min(self.logic_x_range[1], logic_x))
            logic_y = max(self.logic_y_range[0], min(self.logic_y_range[1], logic_y))

            clipped_x = origin_x + logic_x * scale_x
            clipped_y = origin_y - logic_y * scale_y

            self.circles[self.dragging_circle] = QPoint(int(clipped_x), int(clipped_y))
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_circle = None

    def get_circle_positions(self):
        w = self.width()
        h = self.height()
        origin_x = w // 2
        origin_y = h // 2
        scale_x = (self.logic_x_range[1] - self.logic_x_range[0]) / w
        scale_y = (self.logic_y_range[1] - self.logic_y_range[0]) / h

        positions = {}
        for name, pos in self.circles.items():
            lon = (pos.x() - origin_x) * scale_x
            lat = (origin_y - pos.y()) * scale_y
            positions[name] = (lon, lat)
        return positions

class ChorusAudioControl(QWidget):
    def __init__(self, title):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.chorus_files = []
        self.chorus_audio_dict = {}

        self.list_widget = QListWidget()
        self.play_button = QPushButton("Play Chorus")
        self.stop_button = QPushButton("Stop Chorus")
        self.label = QLabel("No chorus audio available.")

        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)
        self.list_widget.itemClicked.connect(self.select_audio)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_selected_path = None

    def add_chorus_audio(self, file_path, audio_segment):
        filename = os.path.basename(file_path)
        self.chorus_files.clear()
        self.chorus_audio_dict.clear()
        self.list_widget.clear()

        self.chorus_files.append(file_path)
        item = QListWidgetItem(f"Chorus: {filename}")
        item.setData(Qt.UserRole, file_path)
        self.list_widget.addItem(item)
        self.chorus_audio_dict[file_path] = audio_segment
        self.label.setText(f"{len(self.chorus_files)} chorus file available.")

    def select_audio(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.isfile(file_path):
            self.current_selected_path = file_path
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def play_audio(self):
        if not self.current_selected_path:
            if self.chorus_files:
                self.current_selected_path = self.chorus_files[-1]
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_selected_path)))
            else:
                self.label.setText("No chorus audio selected.")
                return

        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play Chorus")
        else:
            self.media_player.play()
            self.play_button.setText("Pause Chorus")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play Chorus")

class AudioPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Conversion Tool")
        self.setGeometry(100, 100, 1200, 700)

        self.visualizer = DraggableTargetCircle()
        self.audio_target = TargetAudioControl("Target", self.visualizer)
        self.audio_source = SourceAudioControl("Source")
        self.chorus_control = ChorusAudioControl("Chorus")

        self.converted_player = QMediaPlayer()
        
        self.converted_path = None
        self.converted_files = []
        self.original_converted_files = []

        self.config_form = self.create_config_form()

        self.converted_audio_list = QListWidget()
        self.converted_audio_list.itemClicked.connect(self.play_selected_converted_audio)
        self.volume_sliders = {}
        self.volume_values = {}

        self.chorus_volume_label = QLabel("Chorus Volume: 1.0")
        self.chorus_volume_slider = QSlider(Qt.Horizontal)
        self.chorus_volume_slider.setRange(0, 200)
        self.chorus_volume_slider.setValue(100)
        self.chorus_volume_slider.valueChanged.connect(self.update_chorus_volume)

        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert_and_play)

        self.spatial_button = QPushButton("Apply Spatial Audio")
        self.spatial_button.clicked.connect(self.apply_spatial_audio)
        self.spatial_button.setEnabled(False)

        self.save_button = QPushButton("Save Converted Audio")
        self.save_button.clicked.connect(self.save_audio)
        self.save_button.setEnabled(False)

        self.result_label = QLabel("No output yet.")

        main_layout = QHBoxLayout()

        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.audio_target)
        left_layout.addWidget(self.audio_source)
        left_layout.addWidget(QLabel("<b>Conversion Settings</b>"))
        left_layout.addLayout(self.config_form)
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        left_panel.setWidget(left_widget)

        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>Spatial Positioning</b>"))
        right_layout.addWidget(self.visualizer)
        right_layout.addWidget(QLabel("<b>Converted Audio</b>"))
        right_layout.addWidget(self.converted_audio_list)
        right_layout.addWidget(self.chorus_volume_label)
        right_layout.addWidget(self.chorus_volume_slider)
        right_layout.addWidget(self.chorus_control)
        right_layout.addWidget(self.convert_button)
        right_layout.addWidget(self.spatial_button)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.result_label)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        right_panel.setWidget(right_widget)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.setStyleSheet("""
            QWidget { font-size: 12px; }
            QPushButton { padding: 5px; }
            QLineEdit, QSpinBox, QDoubleSpinBox { max-width: 300px; }
            QListWidget { max-height: 200px; }
        """)

    def create_config_form(self):
        self.source_input = QLineEdit("./examples/source/source_s1.wav")
        self.target_input = QListWidget()
        self.target_input.setSelectionMode(QAbstractItemView.MultiSelection)
        self.target_input.setDragDropMode(QAbstractItemView.InternalMove)

        self.target_audio_paths = []
        self.output_input = QLineEdit("./reconstructed")

        self.diffusion_steps_input = QSpinBox()
        self.diffusion_steps_input.setValue(50)

        self.length_adjust_input = QDoubleSpinBox()
        self.length_adjust_input.setValue(1.0)

        self.inference_cfg_input = QDoubleSpinBox()
        self.inference_cfg_input.setValue(1.0)

        self.f0_condition_input = QCheckBox()
        self.f0_condition_input.setChecked(False)
        self.auto_f0_adjust_input = QCheckBox()
        self.auto_f0_adjust_input.setChecked(False)
        self.fp16_input = QCheckBox()
        self.fp16_input.setChecked(True)

        self.semitone_shift_input = QSpinBox()
        self.semitone_shift_input.setRange(-24, 24)
        self.semitone_shift_input.setValue(0)

        self.checkpoint_input = QLineEdit()
        self.config_input = QLineEdit()

        form = QFormLayout()
        form.addRow("Source Path", self.source_input)
        form.addRow("Target Files", self.target_input)
        form.addRow("Output Dir", self.output_input)
        form.addRow("Diffusion Steps", self.diffusion_steps_input)
        form.addRow("Length Adjust", self.length_adjust_input)
        form.addRow("Inference CFG Rate", self.inference_cfg_input)
        form.addRow("Use F0 Condition", self.f0_condition_input)
        form.addRow("Auto F0 Adjust", self.auto_f0_adjust_input)
        form.addRow("Semitone Shift", self.semitone_shift_input)
        form.addRow("Checkpoint Path", self.checkpoint_input)
        form.addRow("Config Path", self.config_input)
        form.addRow("Use FP16", self.fp16_input)
        return form

    def normalize_audio_power(self, audio):
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        sr = audio.frame_rate

        samples = samples.astype(np.float32) / 32768.0
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(samples)
        target_loudness = -23.0
        normalized_samples = pyln.normalize.loudness(samples, loudness, target_loudness)
        normalized_samples = (normalized_samples * 32768).astype(np.int16)
        normalized_audio = AudioSegment(
            normalized_samples.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        return normalized_audio

    def update_chorus(self):
        if len(self.converted_files) < 1:
            return

        combined = None
        max_len = max(len(audio) for _, _, audio in self.converted_files)
        headroom_db = -6.0  # Increased headroom to prevent clipping

        # Normalize and apply volume adjustments for each audio
        normalized_audios = []
        for file_path, target_filename, audio in self.converted_files:
            volume = self.volume_values.get(file_path, 1.0)
            adjusted_audio = audio + (20 * np.log10(volume))

            if len(adjusted_audio) < max_len:
                adjusted_audio = adjusted_audio + AudioSegment.silent(duration=max_len - len(adjusted_audio))

            samples = np.array(adjusted_audio.get_array_of_samples())
            if adjusted_audio.channels == 2:
                samples = samples.reshape((-1, 2))
            sr = audio.frame_rate

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(samples.mean(axis=1) if samples.ndim > 1 else samples)
            normalized_samples = pyln.normalize.loudness(samples, loudness, -23.0)
            normalized_samples = (normalized_samples * 32768).astype(np.int16)
            normalized_audio = AudioSegment(
                normalized_samples.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=2 if samples.ndim > 1 else 1
            )
            normalized_audios.append(normalized_audio)

        # Combine normalized audios with clipping prevention
        combined = normalized_audios[0]
        for audio in normalized_audios[1:]:
            combined = combined.overlay(audio, gain_during_overlay=-3.0)  # Apply slight gain reduction during overlay

        chorus_volume = self.chorus_volume_slider.value() / 100.0
        combined = combined + (20 * np.log10(chorus_volume))

        if combined.max_dBFS > headroom_db:
            gain_adjust = headroom_db - combined.max_dBFS
            combined = combined.apply_gain(gain_adjust)

        combined_file = os.path.join(tempfile.gettempdir(), "combined_chorus.wav")
        combined.export(combined_file, format="wav")
        self.chorus_control.add_chorus_audio(combined_file, combined)

    def get_args(self):
        source = self.audio_source.current_selected_path or self.source_input.text()
        if not source or not os.path.isfile(source):
            raise ValueError("소스 오디오 파일이 선택되지 않았거나 유효하지 않습니다.")

        checked_targets = []
        for i in range(self.audio_target.list_widget.count()):
            item = self.audio_target.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                filename = item.text()
                target_path = next((path for path in self.audio_target.audio_files if os.path.basename(path) == filename), None)
                if target_path and os.path.isfile(target_path):
                    checked_targets.append(target_path)

        if not checked_targets:
            raise ValueError("체크된 타겟 오디오 파일이 없습니다.")

        return Namespace(
            source=source,
            targets=checked_targets,
            output=self.output_input.text(),
            diffusion_steps=self.diffusion_steps_input.value(),
            length_adjust=self.length_adjust_input.value(),
            inference_cfg_rate=self.inference_cfg_input.value(),
            f0_condition=self.f0_condition_input.isChecked(),
            auto_f0_adjust=self.auto_f0_adjust_input.isChecked(),
            semi_tone_shift=self.semitone_shift_input.value(),
            checkpoint=self.checkpoint_input.text() or None,
            config=self.config_input.text() or None,
            fp16=self.fp16_input.isChecked()
        )

    def apply_brir_convolution(self, audio_data, sr, lon, lat, target_filename):
        try:
            hrir_L, hrir_R, sr_hrir = interpolated_brir('D1_brir', [-lon, lat])
            if sr != sr_hrir:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sr_hrir)
                sr = sr_hrir
            convolved_signal = convolve_hrir(audio_data, hrir_L, hrir_R)
            convolved_file = os.path.join(tempfile.gettempdir(), f"convolved_{target_filename}")
            sf.write(convolved_file, convolved_signal, sr)
            return convolved_file, convolved_signal
        except Exception as e:
            raise RuntimeError(f"컨볼루션 처리 실패 ({target_filename}): {str(e)}")

    def convert_and_play(self):
        from seed_vc import inference
        import glob
        from pydub import AudioSegment

        start_time = time.time()

        try:
            args = self.get_args()
            os.makedirs(args.output, exist_ok=True)
        except ValueError as e:
            self.result_label.setText(str(e))
            return

        try:
            self.result_label.setText("변환 중... 잠시 기다려주세요.")
            self.result_label.repaint()

            self.converted_files = []
            self.original_converted_files = []
            self.converted_audio_list.clear()
            self.volume_sliders.clear()
            self.volume_values.clear()

            for t_path in args.targets:
                args.target = t_path
                target_filename = os.path.basename(t_path)
                name_without_ext = os.path.splitext(target_filename)[0]

                try:
                    inference(args)
                except Exception as e:
                    print(f"{target_filename} 변환 실패: {e}")
                    self.result_label.setText(f"{target_filename} 변환 실패: {str(e)}")
                    continue

                output_files = glob.glob(os.path.join(args.output, "*.wav"))
                if not output_files:
                    self.result_label.setText(f"{target_filename}에 대한 출력 오디오 파일이 없습니다.")
                    continue

                latest_file = max(output_files, key=os.path.getmtime)
                audio = AudioSegment.from_wav(latest_file)
                normalized_audio = self.normalize_audio_power(audio)
                normalized_file = os.path.join(tempfile.gettempdir(), f"normalized_{target_filename}")
                normalized_audio.export(normalized_file, format="wav")

                self.converted_files.append((normalized_file, target_filename, normalized_audio))
                self.original_converted_files.append((normalized_file, target_filename, normalized_audio))

                item = QListWidgetItem(f"변환됨: {target_filename}")
                item.setData(Qt.UserRole, normalized_file)
                self.converted_audio_list.addItem(item)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 200)
                slider.setValue(100)
                slider.valueChanged.connect(lambda value, path=normalized_file: self.update_volume(path, value))
                self.volume_sliders[normalized_file] = slider
                self.volume_values[normalized_file] = 1.0
                slider_item = QListWidgetItem()
                slider_widget = QWidget()
                slider_layout = QHBoxLayout()
                slider_layout.addWidget(QLabel(f"볼륨: {target_filename}"))
                slider_layout.addWidget(slider)
                slider_widget.setLayout(slider_layout)
                slider_item.setSizeHint(slider_widget.sizeHint())
                self.converted_audio_list.addItem(slider_item)
                self.converted_audio_list.setItemWidget(slider_item, slider_widget)

            if not self.converted_files:
                self.result_label.setText("변환된 오디오 파일이 없습니다.")
                return

            end_time = time.time()
            elapsed_time = end_time - start_time

            first_file, first_target, _ = self.converted_files[0]
            self.converted_path = first_file
            self.converted_player.setMedia(QMediaContent(QUrl.fromLocalFile(first_file)))
            self.converted_player.play()
            self.result_label.setText(f"재생 중: 변환된 {first_target} (소요 시간: {elapsed_time:.2f}초)")
            self.spatial_button.setEnabled(True)
            self.save_button.setEnabled(True)

        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.result_label.setText(f"일반 오류: {str(e)} (소요 시간: {elapsed_time:.2f}초)")
            self.spatial_button.setEnabled(False)
            self.save_button.setEnabled(False)

    def apply_spatial_audio(self):
        start_time = time.time()

        try:
            self.result_label.setText("공간 음향 적용 중... 잠시 기다려주세요.")
            self.result_label.repaint()

            circle_positions = self.visualizer.get_circle_positions()
            new_converted_files = []

            self.converted_audio_list.clear()
            self.volume_sliders.clear()
            self.volume_values.clear()

            for normalized_file, target_filename, normalized_audio in self.original_converted_files:
                name_without_ext = os.path.splitext(target_filename)[0]
                if name_without_ext not in circle_positions:
                    self.result_label.setText(f"{target_filename}에 대한 좌표가 없습니다.")
                    continue

                lon, lat = circle_positions[name_without_ext]
                audio_data, sr = librosa.load(normalized_file, sr=None, mono=True)
                convolved_file, convolved_signal = self.apply_brir_convolution(audio_data, sr, lon, lat, target_filename)
                convolved_audio = AudioSegment.from_wav(convolved_file)

                new_converted_files.append((convolved_file, target_filename, convolved_audio))

                item = QListWidgetItem(f"변환됨: {target_filename} (lon: {lon:.2f}, lat: {lat:.2f})")
                item.setData(Qt.UserRole, convolved_file)
                self.converted_audio_list.addItem(item)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 200)
                slider.setValue(100)
                slider.valueChanged.connect(lambda value, path=convolved_file: self.update_volume(path, value))
                self.volume_sliders[convolved_file] = slider
                self.volume_values[convolved_file] = 1.0
                slider_item = QListWidgetItem()
                slider_widget = QWidget()
                slider_layout = QHBoxLayout()
                slider_layout.addWidget(QLabel(f"볼륨: {target_filename}"))
                slider_layout.addWidget(slider)
                slider_widget.setLayout(slider_layout)
                slider_item.setSizeHint(slider_widget.sizeHint())
                self.converted_audio_list.addItem(slider_item)
                self.converted_audio_list.setItemWidget(slider_item, slider_widget)

            if not new_converted_files:
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.result_label.setText(f"공간 음향이 적용된 오디오 파일이 없습니다. (소요 시간: {elapsed_time:.2f}초)")
                return

            self.converted_files = new_converted_files
            self.update_chorus()

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.result_label.setText(f"공간 음향 적용 완료 (소요 시간: {elapsed_time:.2f}초)")

        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.result_label.setText(f"공간 음향 적용 오류: {str(e)} (소요 시간: {elapsed_time:.2f}초)")

    def update_volume(self, file_path, value):
        self.volume_values[file_path] = value / 100.0
        self.update_chorus()

    def update_chorus_volume(self):
        chorus_volume = self.chorus_volume_slider.value() / 100.0
        self.chorus_volume_label.setText(f"코러스 볼륨: {chorus_volume:.2f}")
        self.update_chorus()

    def play_selected_converted_audio(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.isfile(file_path):
            self.converted_path = file_path
            self.converted_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.converted_player.play()
            self.result_label.setText(f"재생 중: {item.text()}")

    def save_audio(self):
        if not self.converted_files and not self.chorus_control.chorus_files:
            self.result_label.setText("저장할 오디오가 없습니다.")
            return

        source_name = os.path.splitext(os.path.basename(self.audio_source.current_selected_path or self.source_input.text()))[0]
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        selected_items = self.converted_audio_list.selectedItems()
        if selected_items:
            for item in selected_items:
                file_path = item.data(Qt.UserRole)
                target_name = os.path.splitext(item.text().replace("변환됨: ", ""))[0]
                params = f"diff{self.diffusion_steps_input.value()}_len{self.length_adjust_input.value()}_cfg{self.inference_cfg_input.value()}"
                if self.f0_condition_input.isChecked():
                    params += "_f0"
                if self.auto_f0_adjust_input.isChecked():
                    params += "_autof0"
                if self.semitone_shift_input.value() != 0:
                    params += f"_semi{self.semitone_shift_input.value()}"
                default_name = f"{source_name}-{target_name}-{params}-{date_str}.wav"
                save_path, _ = QFileDialog.getSaveFileName(self, "변환된 오디오 저장", default_name, "WAV Files (*.wav)")
                if save_path:
                    try:
                        with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
                            dst.write(src.read())
                        self.result_label.setText(f"저장됨: {save_path}")
                    except Exception as e:
                        self.result_label.setText(f"저장 실패: {str(e)}")
        else:
            save_dir = QFileDialog.getExistingDirectory(self, "모든 오디오를 저장할 디렉토리 선택")
            if save_dir:
                try:
                    for file_path, target_name, _ in self.converted_files:
                        params = f"diff{self.diffusion_steps_input.value()}_len{self.length_adjust_input.value()}_cfg{self.inference_cfg_input.value()}"
                        if self.f0_condition_input.isChecked():
                            params += "_f0"
                        if self.auto_f0_adjust_input.isChecked():
                            params += "_autof0"
                        if self.semitone_shift_input.value() != 0:
                            params += f"_semi{self.semitone_shift_input.value()}"
                        base_name = f"{source_name}-{target_name}-{params}-{date_str}.wav"
                        save_path = os.path.join(save_dir, base_name)
                        with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
                            dst.write(src.read())
                    for file_path in self.chorus_control.chorus_files:
                        params = f"diff{self.diffusion_steps_input.value()}_len{self.length_adjust_input.value()}_cfg{self.inference_cfg_input.value()}"
                        if self.f0_condition_input.isChecked():
                            params += "_f0"
                        if self.auto_f0_adjust_input.isChecked():
                            params += "_autof0"
                        if self.semitone_shift_input.value() != 0:
                            params += f"_semi{self.semitone_shift_input.value()}"
                        base_name = f"{source_name}-chorus-{params}-{date_str}.wav"
                        save_path = os.path.join(save_dir, base_name)
                        with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
                            dst.write(src.read())
                    self.result_label.setText(f"모든 오디오가 저장됨: {save_dir}")
                except Exception as e:
                    self.result_label.setText(f"저장 실패: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioPlayer()
    window.show()
    sys.exit(app.exec_())