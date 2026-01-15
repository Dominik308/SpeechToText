import sys
import os
import cv2
import datetime
import uuid
import shutil
import importlib.util
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog, QSplitter,
                             QListWidget, QMessageBox, QDialog, QComboBox, QSlider, QFormLayout, 
                             QDialogButtonBox, QCheckBox, QGroupBox, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QBuffer, QIODevice, QByteArray
from PyQt6.QtGui import QImage, QPixmap, QAction

# Import our custom modules
# NOTE: We avoid importing main.py directly because it has a blocking loop.
# We will replicate the LLM logic here.
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from vector import retriever, add_file_to_db
import database_manager
from ultralytics import YOLO
import threading
import queue
import time

# Constants
MODEL_NAME = "mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso"
YOLO_MODEL = "yolov8n.pt"

class LLMWorker(QThread):
    """
    Runs the LLM inference in a background thread to prevent freezing the GUI.
    """
    response_ready = pyqtSignal(str, str, str, str) # response, context_text, visual_context, session_id

    def __init__(self, start_signal, model_name):
        super().__init__()
        self.model_name = model_name
        self.prompt_template = """
{personality}

Visual Context (what the camera sees): {visual_context}

Document Context: {context}

Question: {question}
"""
        self.llm = None
        self.vlm = None
        self.chain = None
        self._is_ready = False
        
        # We need to store request data
        self.current_question = ""
        self.current_visual_context = ""
        self.current_session_id = ""
        self.current_personality = ""
        self.current_image_b64 = None
        self.current_vision_mode = "LLM + YOLO" # 'LLM + YOLO', 'VLM', 'Hybrid'
        self.current_vlm_model = ""
        self.current_vlm_cpu_mode = False

    def run(self):
        # ---------------- HYBRID (VLM -> LLM) MODE ----------------
        if self.current_vision_mode == 'Hybrid (VLM -> LLM)':
            vlm_desc = ""
            if self.current_image_b64:
                try:
                    # 1. GENERATE DESCRIPTION WITH VLM
                     if not self.vlm or getattr(self.vlm, 'model', '') != self.current_vlm_model:
                         print(f"Initializing VLM (Hybrid) with model: {self.current_vlm_model} | CPU: {self.current_vlm_cpu_mode}")
                         # num_gpu=0 means CPU, num_gpu=99 means max GPU
                         gpu_layers = 0 if self.current_vlm_cpu_mode else 99
                         self.vlm = ChatOllama(model=self.current_vlm_model, num_gpu=gpu_layers)
                    
                     print("Hybrid: Generating image description...")
                     # Simple describe prompt
                     msg = HumanMessage(content=[
                         {"type": "text", "text": "Describe this image in detailed text."},
                         {"type": "image_url", "image_url": f"data:image/jpeg;base64,{self.current_image_b64}"}
                     ])
                     resp = self.vlm.invoke([msg])
                     vlm_desc = f"[VLM Vision]: {resp.content}"
                     print(f"Hybrid Description: {vlm_desc[:50]}...")
                except Exception as e:
                    print(f"Hybrid VLM Error: {e}")
                    vlm_desc = "[Error getting VLM description]"

            # 2. Update visual context and fall through to STANDARD LLM logic
            self.current_visual_context = vlm_desc if vlm_desc else self.current_visual_context

        # ---------------- PURE VLM MODE ----------------
        if self.current_vision_mode == 'VLM (Vision Language Model)':
            try:
                # Initialize VLM if needed
                if not self.vlm or getattr(self.vlm, 'model', '') != self.current_vlm_model:
                     print(f"Initializing VLM ChatOllama with model: {self.current_vlm_model} | CPU: {self.current_vlm_cpu_mode}")
                     # ChatOllama for VLM
                     gpu_layers = 0 if self.current_vlm_cpu_mode else 99
                     self.vlm = ChatOllama(model=self.current_vlm_model, num_gpu=gpu_layers)

                # RAG for VLM
                context_text = ""
                if getattr(self, 'current_use_docs', True) and self.current_question:
                    try:
                        docs = retriever.invoke(self.current_question)
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                    except Exception:
                        pass
                
                final_question = self.current_question
                if context_text:
                    final_question = f"Document Context:\n{context_text}\n\nQuestion: {final_question}"

                # Construct Message
                content_parts = [{"type": "text", "text": final_question}]
                
                # If we have an image
                if self.current_image_b64:
                    content_parts.append({
                        "type": "image_url", 
                        "image_url": f"data:image/jpeg;base64,{self.current_image_b64}"
                    })
                
                # Personality can be prepended to text if needed, or system message
                if self.current_personality:
                     # Prepend personality to text part
                     content_parts[0]["text"] = f"{self.current_personality}\n\n{content_parts[0]['text']}"

                # Invoke
                print("Invoking VLM...")
                msg = HumanMessage(content=content_parts)
                response = self.vlm.invoke([msg])
                
                # response.content is the string
                result_text = response.content
                
                self.response_ready.emit(result_text, context_text, "[VLM Image Processed]", self.current_session_id)
                
            except Exception as e:
                self.response_ready.emit(f"Error during VLM inference: {e}", "", "", "")
            return

        # ---------------- STANDARD LLM MODE ----------------
        # Initialize model in the thread if not already
        if not self.llm:
            try:
                t0 = time.perf_counter()
                # use configured params if present
                llm_kwargs = {}
                temp = getattr(self, 'temperature', None)
                max_toks = getattr(self, 'max_tokens', None)
                if temp is not None:
                    llm_kwargs['temperature'] = float(temp)
                if max_toks is not None:
                    llm_kwargs['max_tokens'] = int(max_toks)
                
                # Force GPU offloading for standard LLM as well
                llm_kwargs['num_gpu'] = 99
                
                print(f"Initializing LLM with params: {llm_kwargs}...")
                self.llm = OllamaLLM(model=self.model_name, **llm_kwargs)
                prompt = ChatPromptTemplate.from_template(self.prompt_template)
                self.chain = prompt | self.llm
                self._is_ready = True
                print(f"LLM initialized in {time.perf_counter()-t0:.2f}s")
            except Exception as e:
                self.response_ready.emit(f"Error initializing LLM: {e}", "", "", "")
                return

        # Perform inference
        if self.current_question:
            try:
                # Optionally retrieve document context (RAG)
                context_text = ""
                if getattr(self, 'current_use_docs', True):
                    try:
                        t_r = time.perf_counter()
                        docs = retriever.invoke(self.current_question)
                        print(f"Retriever took {time.perf_counter()-t_r:.2f}s")
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                    except Exception:
                        context_text = ""

                # Optionally include visual context
                visual_context = self.current_visual_context if getattr(self, 'current_use_visual', True) else ""

                # Run Chain
                t1 = time.perf_counter()
                result = self.chain.invoke({
                    "context": context_text,
                    "question": self.current_question,
                    "visual_context": visual_context,
                    "personality": self.current_personality
                })
                print(f"LLM inference took {time.perf_counter()-t1:.2f}s")

                self.response_ready.emit(result, context_text, visual_context, self.current_session_id)
            except Exception as e:
                self.response_ready.emit(f"Error during inference: {e}", "", "", "")

    def generate(self, question, visual_context, session_id, personality: str = "", 
                 use_docs: bool = True, use_visual: bool = True,
                 image_b64: str = None, vision_mode: str = "LLM + YOLO", vlm_model: str = "llava",
                 vlm_cpu_mode: bool = False):
        self.current_question = question
        self.current_visual_context = visual_context
        self.current_session_id = session_id
        self.current_personality = personality
        self.current_use_docs = bool(use_docs)
        self.current_use_visual = bool(use_visual)
        self.current_image_b64 = image_b64
        self.current_vision_mode = vision_mode
        self.current_vlm_model = vlm_model
        self.current_vlm_cpu_mode = vlm_cpu_mode
        self.start()

class CameraWorker(QThread):
    """
    Captures video frames for preview.
    """
    image_update = pyqtSignal(QImage)

    def __init__(self, frame_skip=2, downscale_width=800):
        super().__init__()
        self.running = True
        self.frame_skip = frame_skip
        self.downscale_width = downscale_width
        # We perform YOLO only on demand now, but keep model loaded if needed elsewhere? 
        # Actually SingleYOLOWorker creates its own model instance which is slow.
        # Better to share the model or keep it here. 
        self.model = YOLO(YOLO_MODEL)
        self.cap = cv2.VideoCapture(0) # Default camera
        self.latest_frame = None

    def run(self):
        counter = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # keep latest frame available for snapshot requests
            self.latest_frame = frame

            # Just show preview
            # We convert every frame for smooth preview
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_update.emit(qt_image.copy())

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()


class SingleYOLOWorker(QThread):
    image_update = pyqtSignal(QImage)
    detections_update = pyqtSignal(str)

    def __init__(self, model, frame):
        super().__init__()
        self.model = model
        self.frame = frame

    def run(self):
        try:
            h, w = self.frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame_resized = cv2.resize(self.frame, (int(w*scale), int(h*scale)))
            else:
                frame_resized = self.frame

            results = self.model(frame_resized, verbose=False)
            annotated_frame = results[0].plot()

            detected_objects = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    detected_objects.append(label)
            from collections import Counter
            counts = Counter(detected_objects)
            desc = ", ".join([f"{count} {obj}" for obj, count in counts.items()]) or "nothing of note"
            self.detections_update.emit(desc)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_update.emit(qt_image.copy())
        except Exception as e:
            print("SingleYOLOWorker error:", e)


class TTSWorker(QThread):
    """Background worker to speak text using pyttsx3 with voice/rate/volume support."""
    def __init__(self, text: str, voice_id: str = None, rate: int = None, volume: float = None):
        super().__init__()
        self.text = text
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume

    def run(self):
        try:
            import pyttsx3
            tts = pyttsx3.init()
            if self.voice_id:
                try:
                    tts.setProperty('voice', self.voice_id)
                except Exception:
                    pass
            if self.rate is not None:
                try:
                    tts.setProperty('rate', int(self.rate))
                except Exception:
                    pass
            if self.volume is not None:
                try:
                    tts.setProperty('volume', float(self.volume))
                except Exception:
                    pass
            tts.say(self.text)
            tts.runAndWait()
        except Exception as e:
            print(f"TTSWorker error: {e}")


class TTSManager:
    """Single long-running pyttsx3 engine with queue to avoid re-initialization cost."""
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._engine = None
        self._thread.start()

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
        return self._engine

    def _run(self):
        while True:
            text, voice_id, rate, volume = self._q.get()
            try:
                engine = self._get_engine()
                if voice_id:
                    try:
                        engine.setProperty('voice', voice_id)
                    except Exception:
                        pass
                if rate is not None:
                    try:
                        engine.setProperty('rate', int(rate))
                    except Exception:
                        pass
                if volume is not None:
                    try:
                        engine.setProperty('volume', float(volume))
                    except Exception:
                        pass
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTSManager error:", e)

    def speak(self, text, voice_id=None, rate=None, volume=None):
        self._q.put((text, voice_id, rate, volume))


def _enumerate_pyttsx3_voices() -> list:
    """Return list of voice dicts: {'id','name','languages'}"""
    results = []
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices') or []
        for v in voices:
            vid = getattr(v, 'id', '') or ''
            name = getattr(v, 'name', '') or vid
            langs = []
            try:
                raw = getattr(v, 'languages', []) or []
                for l in raw:
                    try:
                        if isinstance(l, bytes):
                            langs.append(l.decode('utf-8', errors='ignore'))
                        else:
                            langs.append(str(l))
                    except Exception:
                        langs.append(str(l))
            except Exception:
                pass
            results.append({'id': str(vid), 'name': str(name), 'languages': langs})
    except Exception as e:
        print(f"Failed to enumerate pyttsx3 voices: {e}")
    return results

class VoiceSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Voice Settings')
        self.setModal(True)

        self.form = QFormLayout(self)

        self.lang_cb = QComboBox()
        self.lang_cb.addItems(['Any', 'English', 'German'])

        self.gender_cb = QComboBox()
        self.gender_cb.addItems(['Any', 'Female', 'Male'])

        self.voice_cb = QComboBox()

        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(80, 300)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)

        self.test_btn = QPushButton('Test Voice')
        self.test_btn.clicked.connect(self.on_test)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.form.addRow('Language', self.lang_cb)
        self.form.addRow('Gender', self.gender_cb)
        self.form.addRow('Voice', self.voice_cb)
        self.form.addRow('Rate', self.rate_slider)
        self.form.addRow('Volume', self.volume_slider)
        self.form.addRow('', self.test_btn)
        self.form.addRow('', self.buttons)

        # Enumerate voices first to avoid signals firing before list exists
        self.available_voices = _enumerate_pyttsx3_voices()

        # Connect signals after enumeration
        self.lang_cb.currentTextChanged.connect(self.update_voice_list)
        self.gender_cb.currentTextChanged.connect(self.update_voice_list)

        # Load saved settings
        settings = QSettings('YourOrg', 'SpeechToText')
        lang = settings.value('tts/language', 'Any')
        gender = settings.value('tts/gender', 'Any')
        voice_id = settings.value('tts/voice_id', '')
        rate = int(settings.value('tts/rate', 180))
        volume = int(settings.value('tts/volume', 80))

        self.lang_cb.setCurrentText(lang)
        self.gender_cb.setCurrentText(gender)
        self.rate_slider.setValue(rate)
        self.volume_slider.setValue(volume)

        self.update_voice_list()

        if voice_id:
            idx = self.voice_cb.findData(voice_id)
            if idx >= 0:
                self.voice_cb.setCurrentIndex(idx)

        # Track live test workers so they are not destroyed while running
        self._test_workers = []

    def update_voice_list(self):
        language = self.lang_cb.currentText()
        gender = self.gender_cb.currentText()
        filtered = []

        # safeguard if enumeration failed earlier
        if not getattr(self, 'available_voices', None):
            self.available_voices = _enumerate_pyttsx3_voices()

        for v in self.available_voices:
            name_low = v['name'].lower()
            langs_low = ' '.join(v['languages']).lower()
            vid_low = str(v.get('id', '')).lower()

            lang_ok = True
            if language == 'English':
                lang_ok = ('en' in langs_low) or ('english' in name_low) or ('en' in vid_low)
            elif language == 'German':
                lang_ok = ('de' in langs_low) or ('german' in name_low) or ('de' in vid_low)

            gender_ok = True
            if gender == 'Female':
                gender_ok = ('female' in name_low) or ('frau' in name_low) or any(f in name_low for f in ['anna', 'sara', 'sarah', 'julia', 'maria']) or ('f' in vid_low and 'male' not in vid_low)
            elif gender == 'Male':
                # Enhanced heuristics: check for 'male', common male names, or voice ids that include 'm' markers
                gender_ok = ('male' in name_low) or any(m in name_low for m in ['john', 'david', 'michael', 'peter', 'mark', 'david', 'paul', 'tom', 'thomas']) or (' m' in vid_low) or (vid_low.endswith('m')) or ('-m' in vid_low)

            if lang_ok and gender_ok:
                filtered.append(v)

        # If no voices matched the filters, show all and keep user informed
        if not filtered:
            filtered = self.available_voices

        self.voice_cb.clear()
        for v in filtered:
            display = v['name']
            # Append language hint
            if v['languages']:
                display = f"{display} ({', '.join(v['languages'])})"
            self.voice_cb.addItem(display, v['id'])

    def on_test(self):
        text = "This is a test of the selected voice."
        vid = self.voice_cb.currentData()
        rate = self.rate_slider.value()
        volume = self.volume_slider.value() / 100.0
        worker = TTSWorker(text, voice_id=vid, rate=rate, volume=volume)
        # keep a reference to prevent QThread 'destroyed while running' warnings
        self._test_workers.append(worker)

        def _on_finished():
            try:
                self._test_workers.remove(worker)
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass

        worker.finished.connect(_on_finished)
        worker.start()

def _personality_text_from_mode(mode: str, custom_text: str) -> str:
    if not mode or mode == 'Off':
        return ''
    if mode == 'Default':
        return 'You are a helpful assistant. Use the provided context to answer the user\'s question.'
    if mode == 'Friendly':
        return 'You are a friendly, conversational assistant. Be brief, positive, and helpful.'
    if mode == 'Professional':
        return 'You are a professional assistant. Use formal tone and concise, accurate answers.'
    if mode == 'Custom':
        return custom_text or ''
    return ''

class AssistantSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Assistant Settings')
        self.setModal(True)

        form = QFormLayout(self)
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(['Default', 'Friendly', 'Professional', 'Custom', 'Off'])
        self.custom_edit = QTextEdit()
        self.custom_edit.setPlaceholderText('Enter custom system prompt here...')

        self.vision_system_cb = QComboBox()
        self.vision_system_cb.addItems(['LLM + YOLO', 'VLM (Vision Language Model)', 'Hybrid (VLM -> LLM)'])
        self.vlm_model_edit = QLineEdit()
        self.vlm_model_edit.setPlaceholderText("e.g. llava, moondream, llama3.2-vision")
        self.vlm_cpu_cb = QCheckBox("Run VLM on CPU (Save VRAM)")
        self.vlm_cpu_cb.setToolTip("Useful if running both models simultaneously exceeds GPU memory.")

        self.use_docs_cb = QCheckBox('Include Documents (RAG)')
        self.use_visual_cb = QCheckBox('Include Visual Context')

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form.addRow('Personality Mode', self.mode_cb)
        form.addRow('Custom Prompt', self.custom_edit)
        form.addRow('Vision System', self.vision_system_cb)
        form.addRow('VLM Model Name', self.vlm_model_edit)
        form.addRow('', self.vlm_cpu_cb)
        form.addRow('', self.use_docs_cb)
        form.addRow('', self.use_visual_cb)
        form.addRow('', self.buttons)

        # Load settings
        settings = QSettings('YourOrg', 'SpeechToText')
        mode = settings.value('assistant/personality_mode', 'Default')
        custom = settings.value('assistant/custom_personality', '')
        use_docs = settings.value('assistant/use_docs', True, type=bool)
        use_visual = settings.value('assistant/use_visual', True, type=bool)
        vision_system = settings.value('assistant/vision_system_mode', 'LLM + YOLO')
        vlm_model = settings.value('assistant/vlm_model', 'llava')
        vlm_cpu = settings.value('assistant/vlm_cpu', False, type=bool)

        self.mode_cb.setCurrentText(mode)
        self.custom_edit.setPlainText(custom)
        self.use_docs_cb.setChecked(use_docs)
        self.use_visual_cb.setChecked(use_visual)
        self.vision_system_cb.setCurrentText(vision_system)
        self.vlm_model_edit.setText(vlm_model)
        self.vlm_cpu_cb.setChecked(vlm_cpu)

        # Show/hide custom field
        def on_mode_change(text):
            self.custom_edit.setVisible(text == 'Custom')
        on_mode_change(self.mode_cb.currentText())
        self.mode_cb.currentTextChanged.connect(on_mode_change)

        # Show/hide VLM model field
        def on_vis_system_change(text):
            is_vlm = 'VLM' in text
            self.vlm_model_edit.setEnabled(is_vlm)
            self.vlm_cpu_cb.setEnabled(is_vlm)
        on_vis_system_change(self.vision_system_cb.currentText())
        self.vision_system_cb.currentTextChanged.connect(on_vis_system_change)

    def get_values(self):
        return (self.mode_cb.currentText(),
                self.custom_edit.toPlainText(),
                bool(self.use_docs_cb.isChecked()),
                bool(self.use_visual_cb.isChecked()),
                self.vision_system_cb.currentText(),
                self.vlm_model_edit.text(),
                bool(self.vlm_cpu_cb.isChecked()))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Assistant with Vision & RAG")
        self.setGeometry(100, 100, 1200, 800)

        # Database Init
        database_manager.init_db()
        
        # Session State
        self.session_id = str(uuid.uuid4())
        self.latest_visual_context = "Camera not active"

        # Auto-Describe Timer
        self.auto_timer = QTimer()
        self.auto_timer.interval = 30000 # 30s default, can be adjusted
        self.auto_timer.timeout.connect(self.on_auto_describe_tick)

        # Setup UI
        self.setup_ui()

        # Setup Workers
        self.llm_worker = LLMWorker(None, MODEL_NAME)
        self.llm_worker.response_ready.connect(self.handle_llm_response)
        
        # Load LLM params from settings and assign to worker
        settings = QSettings('YourOrg', 'SpeechToText')
        max_toks = int(settings.value('assistant/max_tokens', 150))
        temp = float(settings.value('assistant/temperature', 0.2))
        self.llm_worker.max_tokens = max_toks
        self.llm_worker.temperature = temp
        # Pre-warm LLM to avoid first-query delay (initialization only)
        QTimer.singleShot(1000, lambda: self.llm_worker.start())

        self.camera_worker = None

        # Try to import mp3_utils from the bundled nvidia ASR app for TTS
        try:
            mp3_path = os.path.join(os.path.dirname(__file__), "nvidia_asr_app-main", "mp3_utils.py")
            spec = importlib.util.spec_from_file_location("nvidia_mp3_utils", mp3_path)
            mp3_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mp3_module)
            self.mp3_utils = mp3_module
            print("Loaded mp3_utils from nvidia_asr_app-main")
        except Exception as e:
            print(f"Could not load mp3_utils: {e}")
            self.mp3_utils = None

        # TTS control
        self.speak_responses = True
        self._tts_workers = []
        self._single_shot_workers = []

        # single TTS manager
        self.tts_manager = TTSManager()

        # Load persisted TTS settings
        settings = QSettings('YourOrg', 'SpeechToText')
        self.tts_language = settings.value('tts/language', 'Any')
        self.tts_gender = settings.value('tts/gender', 'Any')
        self.tts_voice_id = settings.value('tts/voice_id', '')
        self.tts_rate = int(settings.value('tts/rate', 180))
        self.tts_volume = float(settings.value('tts/volume', 80)) / 100.0

        # Load assistant personality settings
        self.assistant_personality_mode = settings.value('assistant/personality_mode', 'Default')
        self.assistant_custom_personality = settings.value('assistant/custom_personality', '')
        # Load whether to include documents/visual context
        self.use_docs = settings.value('assistant/use_docs', True, type=bool)
        self.use_visual = settings.value('assistant/use_visual', True, type=bool)
        
        # Vision System Settings
        self.vision_system_mode = settings.value('assistant/vision_system_mode', 'LLM + YOLO')
        self.vlm_model_name = settings.value('assistant/vlm_model', 'llava')
        self.vlm_cpu_mode = settings.value('assistant/vlm_cpu', False, type=bool)
        self.latest_image_b64 = None

        # compute actual personality text
        self.personality_text = _personality_text_from_mode(self.assistant_personality_mode, self.assistant_custom_personality)

    def setup_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Options')
        
        add_doc_action = QAction('Add Document', self)
        add_doc_action.triggered.connect(self.add_document)
        file_menu.addAction(add_doc_action)
        
        new_session_action = QAction('New Session', self)
        new_session_action.triggered.connect(self.start_new_session)
        file_menu.addAction(new_session_action)

        # Toggle whether assistant should speak responses
        speak_action = QAction('Speak Responses', self, checkable=True)
        speak_action.setChecked(True)
        speak_action.toggled.connect(lambda checked: setattr(self, 'speak_responses', checked))
        file_menu.addAction(speak_action)

        voice_settings_action = QAction('Voice Settings', self)
        voice_settings_action.triggered.connect(self.open_voice_settings)
        file_menu.addAction(voice_settings_action)

        assistant_settings_action = QAction('Assistant Settings', self)
        assistant_settings_action.triggered.connect(self.open_assistant_settings)
        file_menu.addAction(assistant_settings_action)


        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter to resize Chat vs Vision
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: Chat ---
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)
        
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)

        # --- Right Side: Vision ---
        vision_widget = QWidget()
        vision_layout = QVBoxLayout(vision_widget)
        
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #222; color: #aaa; border: 1px solid #444;")
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera_label.setMinimumSize(400, 300)
        
        # New Settings Group Layout
        controls_group = QGroupBox("Vision Controls & Settings")
        grid = QGridLayout()

        # Camera Power
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.stop_cam_btn = QPushButton("Stop Camera")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        
        # Auto Describe
        self.auto_desc_cb = QCheckBox("Auto-Describe (30s)")
        self.auto_desc_cb.toggled.connect(self.toggle_auto_describe)

        # Status
        self.vision_status = QLabel("Detection: None")
        self.vision_status.setWordWrap(True)

        # Grid Placement
        # Row 0: Camera Power
        grid.addWidget(self.start_cam_btn, 0, 0)
        grid.addWidget(self.stop_cam_btn, 0, 1)

        # Row 2: Features
        grid.addWidget(self.auto_desc_cb, 1, 0, 1, 2)
        
        # Row 3: Status
        grid.addWidget(self.vision_status, 2, 0, 1, 2)

        controls_group.setLayout(grid)
        
        vision_layout.addWidget(self.camera_label, stretch=3)
        vision_layout.addWidget(controls_group, stretch=1)

        # Add to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(vision_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def start_camera(self):
        if self.camera_worker is None or not self.camera_worker.isRunning():
            self.camera_worker = CameraWorker()
            self.camera_worker.image_update.connect(self.update_camera_feed)
            # Detections are now demand-based, so no signal connection here for detections
            self.camera_worker.start()
            self.start_cam_btn.setEnabled(False)
            self.stop_cam_btn.setEnabled(True)

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.camera_label.setText("Camera Stopped")
        self.latest_visual_context = "Camera not active"

    # on_yolo_mode_changed removed
    # capture_frame removed

    def _start_single_worker(self, worker):
        # We don't connect image_update/detections_update here generically anymore
        # because the caller might want specific binding (or we bind them in caller)
        # But for 'capture_frame' replacement logic, we want consistent behavior.
        
        # Ensure we keep reference
        self._single_shot_workers.append(worker)
        worker.start()

    def _cleanup_single_worker(self, worker):
        if worker in self._single_shot_workers:
            self._single_shot_workers.remove(worker)
        worker.deleteLater()

    def update_camera_feed(self, image):
        # Scale properly with aspect ratio
        if not self.camera_label.size().isEmpty():
             pixmap = QPixmap.fromImage(image)
             scaled = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
             self.camera_label.setPixmap(scaled)
        
        # Store for VLM usage
        try:
            ba = QByteArray()
            buffer = QBuffer(ba)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            image.save(buffer, "JPG")
            self.latest_image_b64 = ba.toBase64().data().decode('utf-8')
        except Exception:
            pass

    def update_detections(self, desc):
        self.latest_visual_context = desc
        self.vision_status.setText(f"Detection: {desc}")

    def toggle_auto_describe(self, checked):
        if checked:
            self.auto_timer.start(30000) # 30s
            # Trigger immediately only if camera is running
            if self.camera_worker and self.camera_worker.isRunning():
                 self.on_auto_describe_tick()
        else:
            self.auto_timer.stop()

    def _generate_with_active_vision(self, prompt_text, is_auto=False):
        """
        Helper to grab a snapshot, run YOLO (single shot), and then trigger LLM generation.
        Ensures 'only one image' is used per response.
        """
        # Common params
        vision_mode = getattr(self, 'vision_system_mode', 'LLM + YOLO')
        vlm_model = getattr(self, 'vlm_model_name', 'llava')
        vlm_cpu = getattr(self, 'vlm_cpu_mode', False)
        personality = self.personality_text

        # If camera is active, we do the snapshot flow
        if self.camera_worker and self.camera_worker.isRunning() and self.camera_worker.latest_frame is not None:
            # 1. Grab Frame
            frame = self.camera_worker.latest_frame.copy()
            
            # 2. Run YOLO Analysis (Single Shot)
            # We reuse the model loaded in CameraWorker (it's idle since we stopped stream inference)
            worker = SingleYOLOWorker(self.camera_worker.model, frame)
            
            # 3. Connect signals
            # Update UI detection label and internal context
            worker.detections_update.connect(self.update_detections)
            # Update camera feed with the annotated frame so user sees what was captured
            worker.image_update.connect(self.update_camera_feed)
            
            def on_vision_complete():
                self._cleanup_single_worker(worker)
                
                # Now trigger LLM with the updated context/image
                self.llm_worker.generate(
                    prompt_text,
                    self.latest_visual_context,
                    self.session_id,
                    personality=personality,
                    use_docs=self.use_docs,
                    use_visual=True,
                    image_b64=getattr(self, 'latest_image_b64', None),
                    vision_mode=vision_mode,
                    vlm_model=vlm_model,
                    vlm_cpu_mode=vlm_cpu
                )
            
            worker.finished.connect(on_vision_complete)
            self._start_single_worker(worker)
            
            if not is_auto:
                self.chat_display.append("<i>System: Capturing and analyzing scene...</i>")

        else:
            # Fallback if no camera
            self.llm_worker.generate(
                prompt_text,
                self.latest_visual_context,
                self.session_id,
                personality=personality,
                use_docs=self.use_docs,
                use_visual=self.use_visual,
                image_b64=None,
                vision_mode=vision_mode,
                vlm_model=vlm_model,
                vlm_cpu_mode=vlm_cpu
            )

    def on_auto_describe_tick(self):
        if not self.camera_worker or not self.camera_worker.isRunning():
            return
        
        # Don't interrupt if already busy processing a request
        if self.llm_worker.isRunning():
            return

        self.chat_display.append(f"<i>System: Auto-Analyzing Scene...</i>")
        self._generate_with_active_vision("Describe the current scene and what is happening in it.", is_auto=True)

    def open_voice_settings(self):
        dialog = VoiceSettingsDialog(self)
        # PyQt6 uses QDialog.DialogCode.Accepted for exec() return code
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = QSettings('YourOrg', 'SpeechToText')
            language = dialog.lang_cb.currentText()
            gender = dialog.gender_cb.currentText()
            voice_id = dialog.voice_cb.currentData() or ''
            rate = dialog.rate_slider.value()
            volume = dialog.volume_slider.value()
            settings.setValue('tts/language', language)
            settings.setValue('tts/gender', gender)
            settings.setValue('tts/voice_id', voice_id)
            settings.setValue('tts/rate', rate)
            settings.setValue('tts/volume', volume)
            self.tts_language = language
            self.tts_gender = gender
            self.tts_voice_id = voice_id
            self.tts_rate = rate
            self.tts_volume = volume / 100.0
            # Show selected voice in status bar
            try:
                vname = dialog.voice_cb.currentText()
                self.statusBar().showMessage(f"TTS voice set: {vname}")
            except Exception:
                pass

    def open_assistant_settings(self):
        dialog = AssistantSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            mode, custom, use_docs, use_visual, vision_sys, vlm_name, vlm_cpu = dialog.get_values()
            settings = QSettings('YourOrg', 'SpeechToText')
            settings.setValue('assistant/personality_mode', mode)
            settings.setValue('assistant/custom_personality', custom)
            settings.setValue('assistant/use_docs', use_docs)
            settings.setValue('assistant/use_visual', use_visual)
            settings.setValue('assistant/vision_system_mode', vision_sys)
            settings.setValue('assistant/vlm_model', vlm_name)
            settings.setValue('assistant/vlm_cpu', vlm_cpu)

            self.assistant_personality_mode = mode
            self.assistant_custom_personality = custom
            self.use_docs = bool(use_docs)
            self.use_visual = bool(use_visual)
            self.vision_system_mode = vision_sys
            self.vlm_model_name = vlm_name
            self.vlm_cpu_mode = vlm_cpu
            
            self.personality_text = _personality_text_from_mode(mode, custom)
            self.statusBar().showMessage(f"Assistant settings saved. Mode: {vision_sys}")

    def add_document(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document", "", 
                                                   "Documents (*.pdf *.txt *.md *.csv)")
        if file_path:
            # Copy to documents folder for persistence
            filename = os.path.basename(file_path)
            dest_path = os.path.join("documents", filename)
            
            # If checking file existence, etc.
            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(file_path, dest_path)
                
                # Ingest
                self.chat_display.append(f"<i>System: Adding {filename}... please wait.</i>")
                success = add_file_to_db(dest_path)
                if success:
                    self.chat_display.append(f"<i>System: Added {filename} successfully.</i>")
                else:
                    self.chat_display.append(f"<i>System: Failed to add {filename}. Check console.</i>")
            except Exception as e:
                self.chat_display.append(f"<i>System: Error: {e}</i>")

    def start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.chat_display.clear()
        self.chat_display.append("<i>System: New session started.</i>")

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Display user message
        self.chat_display.append(f"<b>You:</b> {text}")
        self.input_field.clear()
        
        # Determine visual context (use latest for now, but will be updated if vision is active)
        vis_context = self.latest_visual_context

        # Determine VLM usage
        vision_mode = getattr(self, 'vision_system_mode', 'LLM + YOLO')
        
        # Save user message to DB
        # Note: We save document_context as "PENDING" or empty for user msg, 
        # but it's really associated with the assistant response.
        database_manager.save_message(self.session_id, "user", text, vis_context)
        
        # Signal loading
        self.chat_display.append("<i>Assistant is thinking...</i>")
        
        # Trigger the generation pipeline (Snapshot -> YOLO -> LLM)
        self._generate_with_active_vision(text)

    def handle_llm_response(self, response, doc_context, visual_context, session_id):
        # Remove "thinking" line approximately (simplification)
        # Better: just append response.
        
        self.chat_display.append(f"<b>Assistant:</b> {response}")
        self.chat_display.append("<hr>")
        
        # Save to DB
        database_manager.save_message(session_id, "assistant", response, visual_context, doc_context)

        # Optionally speak the response using the long-running TTS manager
        if getattr(self, 'speak_responses', False):
            try:
                self.tts_manager.speak(
                    response,
                    voice_id=(self.tts_voice_id or None),
                    rate=(self.tts_rate or None),
                    volume=(self.tts_volume if hasattr(self, 'tts_volume') else None)
                )
            except Exception as e:
                print(f"Failed to enqueue TTS: {e}")
def close_all_tts_workers(workers):
    # wait briefly for each worker to finish, then delete
    for w in list(workers):
        try:
            if w.isRunning():
                w.wait(2000)
        except Exception:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        ret = app.exec()
    finally:
        # ensure background TTS workers are finished before exit
        close_all_tts_workers(window._tts_workers)
    sys.exit(ret)
