import sys
import os
import html
import uuid
import shutil
import queue
import threading
import time
from collections import Counter

import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog, QSplitter,
                             QMessageBox, QDialog, QComboBox, QSlider, QFormLayout, QDialogButtonBox, QCheckBox,
                             QProgressBar, QListWidget, QListWidgetItem, QMenu, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QShortcut, QTextCursor, QTextCharFormat, QTextBlockFormat

# NOTE: We avoid importing main.py directly because it has a blocking loop.
# The LLM logic is replicated in LLMWorker.
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from vector import retriever, add_file_to_db, DOCUMENTS_PATH
import database_manager
import speech_input
from text_utils import SentenceBuffer, clean_for_speech
from ultralytics import YOLO

# Constants
MODEL_NAME = "mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso"
YOLO_MODEL = "yolov8n.pt"
DEFAULT_MAX_TOKENS = 400
DEFAULT_MEMORY_TURNS = 6


def describe_detections(model, results):
    """Turn YOLO results into a short text like '2 person, 1 cup'."""
    labels = []
    for result in results:
        for box in result.boxes:
            labels.append(model.names[int(box.cls[0])])
    counts = Counter(labels)
    return ", ".join(f"{count} {obj}" for obj, count in counts.items()) or "nothing of note"


def frame_to_qimage(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    return QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def downscale(frame, max_width):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


class LLMWorker(QThread):
    """
    Long-running background thread for LLM inference.

    - Requests are queued, so a message sent while the model is still loading or
      answering a previous question is processed afterwards instead of being lost.
    - The answer is streamed chunk by chunk (response_chunk) and finally saved to
      the database and reported via response_done.
    - The previous messages of the session are passed to the model as conversation memory.
    """
    response_chunk = pyqtSignal(int, str, str)        # request_id, session_id, text chunk
    response_done = pyqtSignal(int, str, str, bool)   # request_id, session_id, full text, ok

    SYSTEM_TEMPLATE = """{personality}
Answer in the language of the user's question.

Visual Context (what the camera sees): {visual_context}

Document Context: {context}"""

    def __init__(self, model_name, temperature=0.2, num_predict=DEFAULT_MAX_TOKENS,
                 memory_turns=DEFAULT_MEMORY_TURNS):
        super().__init__()
        self.model_name = model_name
        # These may be changed from the GUI at any time; they apply to the next request.
        self.temperature = temperature
        self.num_predict = num_predict
        self.memory_turns = memory_turns
        self.chain = None
        self._chain_params = None
        self._init_error = ""
        self._requests = queue.Queue()
        self._next_request_id = 0

    def _build_chain(self):
        params = (float(self.temperature), int(self.num_predict))
        if self.chain is not None and params == self._chain_params:
            return
        print(f"Initializing LLM (temperature={params[0]}, num_predict={params[1]})...")
        # num_predict is Ollama's name for the maximum number of generated tokens
        llm = ChatOllama(model=self.model_name, temperature=params[0], num_predict=params[1])
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_TEMPLATE),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        self.chain = prompt | llm
        self._chain_params = params

    def _init_llm(self):
        try:
            self._build_chain()
        except Exception as e:
            self._init_error = str(e)
            print(f"Error initializing LLM: {e}")
            return

        # Pre-warm: make Ollama load the model into (GPU) memory now,
        # so the first real question does not pay the loading time.
        try:
            import ollama
            t0 = time.perf_counter()
            ollama.generate(model=self.model_name, prompt="")
            print(f"LLM model loaded by Ollama in {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"LLM pre-warm skipped: {e}")

    def run(self):
        self._init_llm()
        while True:
            request = self._requests.get()
            if request is None:  # stop signal
                break
            self._process(request)

    def _load_history(self, session_id, message_id):
        if self.memory_turns <= 0 or message_id is None:
            return []
        try:
            rows = database_manager.get_context_history(
                session_id, message_id, max_messages=2 * int(self.memory_turns))
        except Exception as e:
            print(f"Could not load conversation history: {e}")
            return []
        return [HumanMessage(content) if role == 'user' else AIMessage(content) for role, content in rows]

    def _process(self, req):
        rid, session_id = req['request_id'], req['session_id']
        try:
            self._build_chain()
        except Exception as e:
            self.response_done.emit(rid, session_id, f"Error initializing LLM: {e}", False)
            return
        try:
            context_text = ""
            if req['use_docs']:
                try:
                    t_r = time.perf_counter()
                    docs = retriever.invoke(req['question'])
                    print(f"Retriever took {time.perf_counter()-t_r:.2f}s")
                    context_text = "\n\n".join(doc.page_content for doc in docs)
                except Exception as e:
                    print(f"Retriever error (continuing without documents): {e}")

            visual_context = req['visual_context'] if req['use_visual'] else ""
            history = self._load_history(session_id, req['message_id'])

            t1 = time.perf_counter()
            parts = []
            for chunk in self.chain.stream({
                "context": context_text,
                "question": req['question'],
                "visual_context": visual_context,
                "personality": req['personality'],
                "history": history,
            }):
                text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if text:
                    parts.append(text)
                    self.response_chunk.emit(rid, session_id, text)
            answer = "".join(parts).strip()
            print(f"LLM answered in {time.perf_counter()-t1:.2f}s "
                  f"({len(history)} history messages)")

            # Save here (not in the GUI) so the next queued request already sees this answer
            database_manager.save_message(session_id, "assistant", answer, visual_context, context_text,
                                          reply_to=req['message_id'])
            self.response_done.emit(rid, session_id, answer, True)
        except Exception as e:
            self.response_done.emit(rid, session_id, f"Error during inference: {e}", False)

    def generate(self, question, visual_context, session_id, message_id=None, personality: str = "",
                 use_docs: bool = True, use_visual: bool = True):
        """Queue a question. `message_id` is the database id of the stored user message.
        Returns the request id used in the response signals."""
        self._next_request_id += 1
        self._requests.put({
            'request_id': self._next_request_id,
            'question': question,
            'visual_context': visual_context,
            'session_id': session_id,
            'message_id': message_id,
            'personality': personality,
            'use_docs': bool(use_docs),
            'use_visual': bool(use_visual),
        })
        if not self.isRunning():
            self.start()
        return self._next_request_id

    def stop(self, timeout_ms=5000):
        self._requests.put(None)
        self.wait(timeout_ms)


class IngestWorker(QThread):
    """Indexes a document in the background so the GUI does not freeze."""
    ingest_finished = pyqtSignal(str, bool)  # filename, success

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        ok = add_file_to_db(self.file_path)
        self.ingest_finished.emit(os.path.basename(self.file_path), ok)


class SpeechWorker(QThread):
    """Records one utterance from the microphone and transcribes it locally."""
    level_changed = pyqtSignal(float)
    state_changed = pyqtSignal(str)   # 'recording' or 'transcribing'
    text_ready = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, transcriber, device_index=None, language=None, auto_stop=True):
        super().__init__()
        self.transcriber = transcriber
        self.device_index = device_index
        self.language = language
        self.auto_stop = auto_stop
        self._stop_requested = False

    def stop_recording(self):
        self._stop_requested = True

    def run(self):
        # Load the Whisper model while the user is speaking
        threading.Thread(target=self._preload, daemon=True).start()
        try:
            self.state_changed.emit('recording')
            audio = speech_input.record(
                should_stop=lambda: self._stop_requested,
                on_level=self.level_changed.emit,
                device_index=self.device_index,
                auto_stop=self.auto_stop,
            )
            self.state_changed.emit('transcribing')
            self.text_ready.emit(self.transcriber.transcribe(audio, language=self.language))
        except Exception as e:
            self.failed.emit(str(e))

    def _preload(self):
        try:
            self.transcriber.load()
        except Exception as e:
            print(f"Whisper preload failed: {e}")


class CameraWorker(QThread):
    """
    Captures video frames and runs YOLO inference.
    """
    image_update = pyqtSignal(QImage)
    detections_update = pyqtSignal(str)
    camera_error = pyqtSignal(str)

    def __init__(self, model, mode='stream', frame_skip=2, downscale_width=800):
        super().__init__()
        self.running = True
        self.mode = mode  # 'stream' or 'single'
        self.frame_skip = frame_skip
        self.downscale_width = downscale_width
        self.model = model
        self.cap = None
        self.latest_frame = None

    def run(self):
        self.cap = cv2.VideoCapture(0)  # Default camera
        if not self.cap.isOpened():
            self.camera_error.emit("Could not open camera.")
            self.cap.release()
            return

        counter = 0
        failed_reads = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads > 100:
                    self.camera_error.emit("Camera stopped delivering frames.")
                    break
                time.sleep(0.02)
                continue
            failed_reads = 0

            # keep latest frame available for single-capture requests
            self.latest_frame = frame
            counter += 1

            # In 'single' mode we only show previews; detection happens on "Capture Frame".
            # In 'stream' mode detection is throttled to every n-th frame.
            if self.mode == 'single' or counter % self.frame_skip != 0:
                self.image_update.emit(frame_to_qimage(frame))
                continue

            results = self.model(downscale(frame, self.downscale_width), verbose=False)
            self.detections_update.emit(describe_detections(self.model, results))
            self.image_update.emit(frame_to_qimage(results[0].plot()))

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


class SingleYOLOWorker(QThread):
    image_update = pyqtSignal(QImage)
    detections_update = pyqtSignal(str)

    def __init__(self, model, frame):
        super().__init__()
        self.model = model
        self.frame = frame

    def run(self):
        try:
            results = self.model(downscale(self.frame, 800), verbose=False)
            self.detections_update.emit(describe_detections(self.model, results))
            self.image_update.emit(frame_to_qimage(results[0].plot()))
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

    def clear(self):
        """Drop everything that is still waiting to be spoken."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


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

        self.use_docs_cb = QCheckBox('Include Documents (RAG)')
        self.use_visual_cb = QCheckBox('Include Visual Context')

        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(0, 30)
        self.memory_spin.setSuffix(' exchanges')
        self.memory_spin.setToolTip('How many previous question/answer pairs the assistant remembers (0 = none)')
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4096)
        self.max_tokens_spin.setSingleStep(50)
        self.max_tokens_spin.setSuffix(' tokens')

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form.addRow('Personality Mode', self.mode_cb)
        form.addRow('Custom Prompt', self.custom_edit)
        form.addRow('', self.use_docs_cb)
        form.addRow('', self.use_visual_cb)
        form.addRow('Conversation Memory', self.memory_spin)
        form.addRow('Max Answer Length', self.max_tokens_spin)
        form.addRow('', self.buttons)

        # Load settings
        settings = QSettings('YourOrg', 'SpeechToText')
        mode = settings.value('assistant/personality_mode', 'Default')
        custom = settings.value('assistant/custom_personality', '')
        use_docs = settings.value('assistant/use_docs', True, type=bool)
        use_visual = settings.value('assistant/use_visual', True, type=bool)
        self.mode_cb.setCurrentText(mode)
        self.custom_edit.setPlainText(custom)
        self.use_docs_cb.setChecked(use_docs)
        self.use_visual_cb.setChecked(use_visual)
        self.memory_spin.setValue(int(settings.value('assistant/memory_turns', DEFAULT_MEMORY_TURNS)))
        self.max_tokens_spin.setValue(int(settings.value('assistant/max_tokens', DEFAULT_MAX_TOKENS)))

        # Show/hide custom field
        def on_mode_change(text):
            self.custom_edit.setVisible(text == 'Custom')
        on_mode_change(self.mode_cb.currentText())
        self.mode_cb.currentTextChanged.connect(on_mode_change)

    def get_values(self):
        return (self.mode_cb.currentText(),
                self.custom_edit.toPlainText(),
                bool(self.use_docs_cb.isChecked()),
                bool(self.use_visual_cb.isChecked()),
                self.memory_spin.value(),
                self.max_tokens_spin.value())

class SpeechSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Speech Input Settings')
        self.setModal(True)
        form = QFormLayout(self)

        self.mic_cb = QComboBox()
        self.mic_cb.addItem('System Default', '')
        try:
            for _idx, name in speech_input.list_input_devices():
                self.mic_cb.addItem(name, name)
        except Exception as e:
            self.mic_cb.addItem(f'(could not list devices: {e})', '')

        self.model_cb = QComboBox()
        self.model_cb.addItems(speech_input.MODEL_SIZES)
        self.language_cb = QComboBox()
        self.language_cb.addItems(list(speech_input.LANGUAGES))
        self.device_cb = QComboBox()
        self.device_cb.addItem('Auto (GPU if available)', 'auto')
        self.device_cb.addItem('GPU (CUDA)', 'cuda')
        self.device_cb.addItem('CPU', 'cpu')
        self.auto_stop_cb = QCheckBox('Stop recording automatically after a pause')
        self.auto_send_cb = QCheckBox('Send recognized text immediately')

        hint = QLabel('Larger models are more accurate but slower. '
                      'They are downloaded automatically on first use.')
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form.addRow('Microphone', self.mic_cb)
        form.addRow('Whisper Model', self.model_cb)
        form.addRow('Language', self.language_cb)
        form.addRow('Compute Device', self.device_cb)
        form.addRow('', self.auto_stop_cb)
        form.addRow('', self.auto_send_cb)
        form.addRow('', hint)
        form.addRow('', buttons)

        cfg = load_speech_settings()
        idx = self.mic_cb.findData(cfg['mic_name'])
        self.mic_cb.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_cb.setCurrentText(cfg['model_size'])
        self.language_cb.setCurrentText(cfg['language'])
        idx = self.device_cb.findData(cfg['compute_device'])
        self.device_cb.setCurrentIndex(idx if idx >= 0 else 0)
        self.auto_stop_cb.setChecked(cfg['auto_stop'])
        self.auto_send_cb.setChecked(cfg['auto_send'])

    def get_values(self):
        return {
            'mic_name': self.mic_cb.currentData() or '',
            'model_size': self.model_cb.currentText(),
            'language': self.language_cb.currentText(),
            'compute_device': self.device_cb.currentData() or 'auto',
            'auto_stop': self.auto_stop_cb.isChecked(),
            'auto_send': self.auto_send_cb.isChecked(),
        }


def load_speech_settings():
    settings = QSettings('YourOrg', 'SpeechToText')
    return {
        'mic_name': settings.value('stt/mic_name', ''),
        'model_size': settings.value('stt/model_size', speech_input.DEFAULT_MODEL_SIZE),
        'language': settings.value('stt/language', 'Auto'),
        'compute_device': settings.value('stt/compute_device', 'auto'),
        'auto_stop': settings.value('stt/auto_stop', True, type=bool),
        'auto_send': settings.value('stt/auto_send', True, type=bool),
    }


def save_speech_settings(values):
    settings = QSettings('YourOrg', 'SpeechToText')
    for key, value in values.items():
        settings.setValue(f'stt/{key}', value)


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

        # Setup UI
        self.setup_ui()

        # Setup Workers
        settings = QSettings('YourOrg', 'SpeechToText')
        self.llm_worker = LLMWorker(
            MODEL_NAME,
            temperature=float(settings.value('assistant/temperature', 0.2)),
            num_predict=int(settings.value('assistant/max_tokens', DEFAULT_MAX_TOKENS)),
            memory_turns=int(settings.value('assistant/memory_turns', DEFAULT_MEMORY_TURNS)),
        )
        self.llm_worker.response_chunk.connect(self.on_response_chunk)
        self.llm_worker.response_done.connect(self.on_response_done)
        self.pending_requests = 0
        # Answer currently being streamed into the chat: {'rid', 'cursor', 'sentences'}
        self._stream = None
        # Pre-warm LLM to avoid first-query delay
        QTimer.singleShot(1000, self.llm_worker.start)

        self.camera_worker = None
        self._yolo_model = None
        # Keep references to short-lived worker threads until they finish
        self._bg_workers = []

        # TTS control
        self.speak_responses = True
        self.tts_manager = TTSManager()

        # Speech input (STT)
        self.speech_worker = None
        self._transcriber = None

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
        # compute actual personality text
        self.personality_text = _personality_text_from_mode(self.assistant_personality_mode, self.assistant_custom_personality)

        self.refresh_session_list()

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

        speech_settings_action = QAction('Speech Input Settings', self)
        speech_settings_action.triggered.connect(self.open_speech_settings)
        file_menu.addAction(speech_settings_action)


        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter to resize Sessions vs Chat vs Vision
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Far left: Sessions ---
        sessions_widget = QWidget()
        sessions_layout = QVBoxLayout(sessions_widget)
        sessions_layout.setContentsMargins(0, 0, 0, 0)
        new_session_btn = QPushButton("New Session")
        new_session_btn.clicked.connect(self.start_new_session)
        self.session_list = QListWidget()
        self.session_list.setWordWrap(True)
        self.session_list.setStyleSheet("QListWidget::item { padding: 6px 4px; border-bottom: 1px solid palette(midlight); }")
        self.session_list.itemClicked.connect(self.on_session_clicked)
        self.session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self.on_session_context_menu)
        sessions_layout.addWidget(new_session_btn)
        sessions_layout.addWidget(self.session_list)

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

        self.mic_btn = QPushButton("Speak")
        self.mic_btn.setToolTip("Speak your message (Ctrl+Space)")
        self.mic_btn.clicked.connect(self.toggle_recording)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.toggle_recording)

        self.mic_level = QProgressBar()
        self.mic_level.setRange(0, 100)
        self.mic_level.setTextVisible(False)
        self.mic_level.setFixedWidth(80)
        self.mic_level.setVisible(False)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.mic_level)
        input_layout.addWidget(self.mic_btn)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)

        # --- Right Side: Vision ---
        vision_widget = QWidget()
        vision_layout = QVBoxLayout(vision_widget)
        
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.camera_label.setMinimumSize(400, 300)
        
        self.vision_status = QLabel("Current Detection: None")
        
        cam_controls = QHBoxLayout()
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.stop_cam_btn = QPushButton("Stop Camera")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)

        self.yolo_mode_cb = QComboBox()
        self.yolo_mode_cb.addItems(['Stream', 'Single'])
        self.yolo_mode_cb.currentTextChanged.connect(self.on_yolo_mode_changed)
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_frame)

        cam_controls.addWidget(self.start_cam_btn)
        cam_controls.addWidget(self.stop_cam_btn)
        cam_controls.addWidget(self.yolo_mode_cb)
        cam_controls.addWidget(self.capture_btn)
        
        vision_layout.addWidget(self.camera_label)
        vision_layout.addWidget(self.vision_status)
        vision_layout.addLayout(cam_controls)
        vision_layout.addStretch()

        # Add to splitter
        splitter.addWidget(sessions_widget)
        splitter.addWidget(chat_widget)
        splitter.addWidget(vision_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 600, 380])

        main_layout.addWidget(splitter)

    @property
    def yolo_model(self):
        """YOLO model, loaded once on first use and shared by all camera workers."""
        if self._yolo_model is None:
            self._yolo_model = YOLO(YOLO_MODEL)
        return self._yolo_model

    def _start_bg_worker(self, worker):
        """Start a short-lived QThread and keep a reference until it has finished."""
        self._bg_workers.append(worker)

        def _cleanup():
            if worker in self._bg_workers:
                self._bg_workers.remove(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup)
        worker.start()

    def start_camera(self):
        mode = 'stream' if self.yolo_mode_cb.currentText() == 'Stream' else 'single'
        if self.camera_worker is None or not self.camera_worker.isRunning():
            self.camera_worker = CameraWorker(self.yolo_model, mode=mode)
            self.camera_worker.image_update.connect(self.update_camera_feed)
            self.camera_worker.detections_update.connect(self.update_detections)
            self.camera_worker.camera_error.connect(self.on_camera_error)
            self.camera_worker.start()
            self.start_cam_btn.setEnabled(False)
            self.stop_cam_btn.setEnabled(True)
            self.capture_btn.setEnabled(mode == 'single')

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.capture_btn.setEnabled(self.yolo_mode_cb.currentText() == 'Single')
        self.camera_label.setText("Camera Stopped")
        self.latest_visual_context = "Camera not active"
        self.vision_status.setText("Current Detection: None")

    def on_camera_error(self, message):
        self.stop_camera()
        self.camera_label.setText(message)
        self.statusBar().showMessage(message, 5000)

    def on_yolo_mode_changed(self, text):
        if self.camera_worker:
            self.camera_worker.mode = 'stream' if text == 'Stream' else 'single'
        self.capture_btn.setEnabled(text == 'Single')

    def capture_frame(self):
        # if camera running, use latest frame
        if self.camera_worker and self.camera_worker.latest_frame is not None:
            frame = self.camera_worker.latest_frame.copy()
        else:
            # grab one frame with temporary capture
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self, "Capture Failed", "Could not capture frame from camera.")
                return
        worker = SingleYOLOWorker(self.yolo_model, frame)
        worker.image_update.connect(self.update_camera_feed)
        worker.detections_update.connect(self.update_detections)
        self._start_bg_worker(worker)

    def update_camera_feed(self, image):
        self.camera_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.camera_label.width(), self.camera_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_detections(self, desc):
        self.latest_visual_context = desc
        self.vision_status.setText(f"Seeing: {desc}")

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
            mode, custom, use_docs, use_visual, memory_turns, max_tokens = dialog.get_values()
            settings = QSettings('YourOrg', 'SpeechToText')
            settings.setValue('assistant/personality_mode', mode)
            settings.setValue('assistant/custom_personality', custom)
            settings.setValue('assistant/use_docs', use_docs)
            settings.setValue('assistant/use_visual', use_visual)
            settings.setValue('assistant/memory_turns', memory_turns)
            settings.setValue('assistant/max_tokens', max_tokens)
            self.llm_worker.memory_turns = memory_turns
            self.llm_worker.num_predict = max_tokens
            self.assistant_personality_mode = mode
            self.assistant_custom_personality = custom
            self.use_docs = bool(use_docs)
            self.use_visual = bool(use_visual)
            self.personality_text = _personality_text_from_mode(mode, custom)
            self.statusBar().showMessage(f"Assistant personality set: {mode}")

    # ------------------------------------------------------------------ speech input

    def open_speech_settings(self):
        dialog = SpeechSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.get_values()
            save_speech_settings(values)
            self.statusBar().showMessage(
                f"Speech input: {values['model_size']} model, language {values['language']}", 5000)

    def _get_transcriber(self, cfg):
        """Reuse the loaded Whisper model unless model or device changed."""
        t = self._transcriber
        if t is None or t.model_size != cfg['model_size'] or t.device != cfg['compute_device']:
            self._transcriber = speech_input.Transcriber(cfg['model_size'], cfg['compute_device'])
        return self._transcriber

    @staticmethod
    def _resolve_mic_index(mic_name):
        if not mic_name:
            return None
        try:
            for idx, name in speech_input.list_input_devices():
                if name == mic_name:
                    return idx
        except Exception as e:
            print(f"Could not list audio devices: {e}")
        print(f"Microphone '{mic_name}' not found, using system default.")
        return None

    def toggle_recording(self):
        if self.speech_worker is not None:
            # Second press while recording: stop and transcribe
            self.speech_worker.stop_recording()
            return

        # Don't let queued speech output talk over the user
        self.tts_manager.clear()
        cfg = load_speech_settings()
        self.speech_worker = SpeechWorker(
            self._get_transcriber(cfg),
            device_index=self._resolve_mic_index(cfg['mic_name']),
            language=speech_input.LANGUAGES.get(cfg['language']),
            auto_stop=cfg['auto_stop'],
        )
        self._speech_auto_send = cfg['auto_send']
        self.speech_worker.level_changed.connect(self.on_mic_level)
        self.speech_worker.state_changed.connect(self.on_speech_state)
        self.speech_worker.text_ready.connect(self.on_speech_text)
        self.speech_worker.failed.connect(self.on_speech_failed)
        self.speech_worker.finished.connect(self.on_speech_finished)
        self.speech_worker.start()

    def on_mic_level(self, level):
        # RMS of speech is roughly 0.01-0.3; show it on a log-like scale
        self.mic_level.setValue(min(100, int((level ** 0.5) * 180)))

    def on_speech_state(self, state):
        if state == 'recording':
            self.mic_btn.setText("Stop")
            self.mic_level.setValue(0)
            self.mic_level.setVisible(True)
            self.statusBar().showMessage("Listening... (press Stop or Ctrl+Space when done)")
        elif state == 'transcribing':
            self.mic_btn.setText("...")
            self.mic_btn.setEnabled(False)
            self.mic_level.setVisible(False)
            self.statusBar().showMessage("Transcribing...")

    def on_speech_text(self, text):
        self.statusBar().clearMessage()
        if not text:
            self.statusBar().showMessage("No speech recognized.", 4000)
            return
        current = self.input_field.text().strip()
        self.input_field.setText(f"{current} {text}".strip())
        if self._speech_auto_send:
            self.send_message()
        else:
            self.input_field.setFocus()

    def on_speech_failed(self, message):
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Speech Input", f"Speech input failed:\n{message}")

    def on_speech_finished(self):
        if self.speech_worker is not None:
            self.speech_worker.deleteLater()
        self.speech_worker = None
        self.mic_btn.setText("Speak")
        self.mic_btn.setEnabled(True)
        self.mic_level.setVisible(False)
        if self.pending_requests:
            self._update_thinking_status()

    # ------------------------------------------------------------------ documents

    def add_document(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document", "",
                                                   "Documents (*.pdf *.txt *.md *.csv)")
        if not file_path:
            return
        # Copy to documents folder for persistence (overwrites an older file with the same name)
        filename = os.path.basename(file_path)
        dest_path = os.path.join(DOCUMENTS_PATH, filename)
        try:
            os.makedirs(DOCUMENTS_PATH, exist_ok=True)
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                shutil.copy2(file_path, dest_path)
        except Exception as e:
            self.append_system_message(f"Error copying {filename}: {e}")
            return

        self.append_system_message(f"Adding {filename}... please wait.")
        worker = IngestWorker(dest_path)
        worker.ingest_finished.connect(self.on_ingest_finished)
        self._start_bg_worker(worker)

    def on_ingest_finished(self, filename, success):
        if success:
            self.append_system_message(f"Added {filename} successfully.")
        else:
            self.append_system_message(f"Failed to add {filename}. Check console.")

    def append_system_message(self, text):
        self.chat_display.append(f"<i>System: {html.escape(text)}</i>")

    # ------------------------------------------------------------------ sessions

    def refresh_session_list(self):
        try:
            sessions = database_manager.get_all_sessions()
        except Exception as e:
            print(f"Could not load sessions: {e}")
            return
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session_id, start_time, title in sessions:
            title = (title or "(no messages)").strip().replace("\n", " ")
            if len(title) > 60:
                title = title[:57] + "..."
            date = str(start_time or "")[:16]
            item = QListWidgetItem(f"{title}\n{date}")
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            item.setToolTip(title)
            self.session_list.addItem(item)
            if session_id == self.session_id:
                item.setSelected(True)
                self.session_list.setCurrentItem(item)
        self.session_list.blockSignals(False)

    def _reset_chat_view(self):
        self.tts_manager.clear()
        self._stream = None
        self.chat_display.clear()

    def start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self._reset_chat_view()
        self.append_system_message("New session started.")
        self.session_list.clearSelection()

    def on_session_clicked(self, item):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id != self.session_id:
            self.load_session(session_id)

    def load_session(self, session_id):
        self.session_id = session_id
        self._reset_chat_view()
        for role, content in database_manager.get_session_history(session_id):
            if role == 'user':
                self.append_user_message(content)
            else:
                self.append_assistant_message(content)
        self.refresh_session_list()

    def on_session_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Session")
        if menu.exec(self.session_list.mapToGlobal(pos)) == delete_action:
            session_id = item.data(Qt.ItemDataRole.UserRole)
            answer = QMessageBox.question(self, "Delete Session",
                                          "Delete this conversation permanently?")
            if answer == QMessageBox.StandardButton.Yes:
                database_manager.delete_session(session_id)
                if session_id == self.session_id:
                    self.start_new_session()
                self.refresh_session_list()

    # ------------------------------------------------------------------ chat

    def _scroll_chat_to_bottom(self):
        bar = self.chat_display.verticalScrollBar()
        bar.setValue(bar.maximum())

    def append_user_message(self, text):
        self.chat_display.append(f"<b>You:</b> {html.escape(text)}")
        self._scroll_chat_to_bottom()

    def append_assistant_message(self, text, error=False):
        if error:
            body = html.escape(text).replace("\n", "<br>")
            self.chat_display.append(f"<b>Assistant:</b> <span style='color:#c0392b'>{body}</span>")
        else:
            self.chat_display.append("<b>Assistant:</b>")
            cursor = QTextCursor(self.chat_display.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
            self._insert_markdown(cursor, text)
        self._scroll_chat_to_bottom()

    @staticmethod
    def _insert_markdown(cursor, text):
        # Keep single line breaks of the model's answer (Markdown would join the lines)
        cursor.insertMarkdown(text.replace("\n", "  \n"))

    def _update_thinking_status(self):
        if self.pending_requests > 0:
            self.statusBar().showMessage(f"Assistant is thinking... ({self.pending_requests} pending)")
        else:
            self.statusBar().clearMessage()

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.append_user_message(text)
        self.input_field.clear()

        vis_context = self.latest_visual_context
        is_new_session = not database_manager.get_session_history(self.session_id)
        message_id = database_manager.save_message(self.session_id, "user", text, vis_context)
        if is_new_session:
            self.refresh_session_list()

        self.pending_requests += 1
        self._update_thinking_status()

        self.llm_worker.generate(
            text,
            vis_context,
            self.session_id,
            message_id=message_id,
            personality=self.personality_text,
            use_docs=self.use_docs,
            use_visual=self.use_visual
        )

    def _speak(self, text):
        text = clean_for_speech(text)
        if not (self.speak_responses and text):
            return
        try:
            self.tts_manager.speak(
                text,
                voice_id=(self.tts_voice_id or None),
                rate=(self.tts_rate or None),
                volume=self.tts_volume
            )
        except Exception as e:
            print(f"Failed to enqueue TTS: {e}")

    def on_response_chunk(self, request_id, session_id, text):
        if session_id != self.session_id:
            return  # answer for a session that is not shown
        if self._stream is None or self._stream['rid'] != request_id:
            # First chunk: open a new "Assistant:" paragraph and keep a cursor to it.
            # The cursor stays inside this paragraph even if the user sends another
            # message meanwhile (which is appended below it).
            self.chat_display.append("<b>Assistant:</b>")
            cursor = QTextCursor(self.chat_display.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
            self._stream = {'rid': request_id, 'cursor': cursor, 'start': cursor.position(),
                            'sentences': SentenceBuffer()}
            text = text.lstrip()
        bar = self.chat_display.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self._stream['cursor'].insertText(text, QTextCharFormat())
        if at_bottom:
            self._scroll_chat_to_bottom()
        for sentence in self._stream['sentences'].feed(text):
            self._speak(sentence)

    def on_response_done(self, request_id, session_id, text, ok):
        self.pending_requests = max(0, self.pending_requests - 1)
        self._update_thinking_status()
        self.refresh_session_list()

        if session_id != self.session_id:
            return
        stream = self._stream if self._stream and self._stream['rid'] == request_id else None
        self._stream = None
        if not ok:
            # Error: show it (below a partial answer, if any); errors are not spoken or saved
            self.append_assistant_message(text, error=True)
        elif stream is None:
            # Nothing was streamed (e.g. empty answer): show the full text at once
            self.append_assistant_message(text)
            self._speak(text)
        else:
            # Replace the streamed plain text with the rendered Markdown version
            cursor = QTextCursor(self.chat_display.document())
            cursor.setPosition(stream['start'])
            cursor.setPosition(stream['cursor'].position(), QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self._insert_markdown(cursor, text)
            self._speak(stream['sentences'].flush())

    def closeEvent(self, event):
        # Release camera and stop background threads before the window closes
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()
            self.speech_worker.wait(5000)
        self.llm_worker.stop()
        for worker in list(self._bg_workers):
            worker.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
