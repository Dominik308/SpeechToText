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
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QShortcut, QTextCursor, QTextCharFormat, QTextBlockFormat

# NOTE: We avoid importing main.py directly because it has a blocking loop.
# The LLM logic is replicated in LLMWorker.
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from vector import retriever, add_file_to_db, DOCUMENTS_PATH
import database_manager
import speech_input
from text_utils import SentenceBuffer, prepare_for_speech
import tts_engine
from vision_labels_de import label_de
from ultralytics import YOLO

# Constants
MODEL_NAME = "mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso"
YOLO_MODEL = "yolov8n.pt"
DEFAULT_MAX_TOKENS = 400
DEFAULT_MEMORY_TURNS = 6
CAMERA_OFF_CONTEXT = "Kamera ist aus"

# Answer language of the assistant
ANSWER_LANGUAGES = [("de", "Immer Deutsch (empfohlen)"), ("auto", "In der Sprache der Frage")]
LANGUAGE_INSTRUCTIONS = {
    "de": ("Antworte immer auf Deutsch – in natürlichem, korrektem und gut verständlichem Deutsch, "
           "auch wenn Dokumente oder Kamera-Informationen englisch sind."),
    "auto": "Antworte in der Sprache, in der die Frage gestellt wurde.",
}


def describe_detections(model, results):
    """Turn YOLO results into a short German text like '2× Person, 1× Tasse'."""
    labels = []
    for result in results:
        for box in result.boxes:
            labels.append(label_de(model.names[int(box.cls[0])]))
    counts = Counter(labels)
    return ", ".join(f"{count}× {obj}" for obj, count in counts.items()) or "nichts Besonderes"


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
{language_instruction}
Deine Antworten werden oft vorgelesen: Formuliere klar, in ganzen Sätzen und ohne Tabellen.

Was die Kamera gerade sieht: {visual_context}

Auszüge aus den Dokumenten des Nutzers (nur verwenden, wenn sie zur Frage passen):
{context}"""

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
        params = (self.model_name, float(self.temperature), int(self.num_predict))
        if self.chain is not None and params == self._chain_params:
            return
        print(f"Initializing LLM {params[0]} (temperature={params[1]}, num_predict={params[2]})...")
        # num_predict is Ollama's name for the maximum number of generated tokens
        llm = ChatOllama(model=params[0], temperature=params[1], num_predict=params[2])
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
            self.response_done.emit(rid, session_id, f"Fehler beim Starten des Sprachmodells: {e}", False)
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

            visual_context = req['visual_context'] if req['use_visual'] else "(Kamera nicht berücksichtigt)"
            history = self._load_history(session_id, req['message_id'])

            t1 = time.perf_counter()
            parts = []
            for chunk in self.chain.stream({
                "context": context_text,
                "question": req['question'],
                "visual_context": visual_context,
                "personality": req['personality'],
                "language_instruction": LANGUAGE_INSTRUCTIONS.get(req['answer_language'],
                                                                  LANGUAGE_INSTRUCTIONS['de']),
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
            self.response_done.emit(rid, session_id, f"Fehler bei der Antwort des Sprachmodells: {e}", False)

    def generate(self, question, visual_context, session_id, message_id=None, personality: str = "",
                 use_docs: bool = True, use_visual: bool = True, answer_language: str = "de"):
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
            'answer_language': answer_language,
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
            self.camera_error.emit("Kamera konnte nicht geöffnet werden.")
            self.cap.release()
            return

        counter = 0
        failed_reads = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads > 100:
                    self.camera_error.emit("Die Kamera liefert keine Bilder mehr.")
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


class StatusBridge(QObject):
    """Forwards status messages from background threads to the GUI thread."""
    message = pyqtSignal(str)


def load_tts_settings():
    settings = QSettings('YourOrg', 'SpeechToText')
    return tts_engine.TTSSettings(
        engine=settings.value('tts/engine', tts_engine.ENGINE_PIPER),
        piper_voice=settings.value('tts/piper_voice', tts_engine.DEFAULT_VOICE),
        system_voice_id=settings.value('tts/voice_id', ''),
        speed=int(settings.value('tts/speed', 100)) / 100.0,
        volume=int(settings.value('tts/volume', 80)) / 100.0,
    )


def save_tts_settings(tts):
    settings = QSettings('YourOrg', 'SpeechToText')
    settings.setValue('tts/engine', tts.engine)
    settings.setValue('tts/piper_voice', tts.piper_voice)
    settings.setValue('tts/voice_id', tts.system_voice_id)
    settings.setValue('tts/speed', int(round(tts.speed * 100)))
    settings.setValue('tts/volume', int(round(tts.volume * 100)))


class VoiceSettingsDialog(QDialog):
    TEST_TEXT = ("Hallo! So klinge ich. Bei ca. 21 °C ist es z. B. ideal für einen Spaziergang, "
                 "und am 3. Oktober ist Feiertag.")

    def __init__(self, speech_output, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sprachausgabe')
        self.setModal(True)
        self.speech_output = speech_output
        form = QFormLayout(self)

        self.engine_cb = QComboBox()
        self.engine_cb.addItem('Piper – natürliche deutsche Stimme (empfohlen)', tts_engine.ENGINE_PIPER)
        self.engine_cb.addItem('System-Stimme (Windows)', tts_engine.ENGINE_SYSTEM)

        self.piper_cb = QComboBox()
        for key, label in tts_engine.GERMAN_VOICES:
            self.piper_cb.addItem(label, key)

        self.system_cb = QComboBox()
        for vid, name, is_german in tts_engine.list_system_voices():
            self.system_cb.addItem(f"{name} (Deutsch)" if is_german else name, vid)
        if self.system_cb.count() == 0:
            self.system_cb.addItem('(keine System-Stimmen gefunden)', '')

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_label = QLabel()
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} %"))
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)

        test_btn = QPushButton('Stimme testen')
        test_btn.clicked.connect(self.on_test)

        hint = QLabel('Die Piper-Stimme läuft lokal und wird beim ersten Benutzen einmalig heruntergeladen. '
                      'Mit Esc lässt sich das Vorlesen jederzeit abbrechen.')
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Abbrechen')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form.addRow('Sprachausgabe', self.engine_cb)
        form.addRow('Piper-Stimme', self.piper_cb)
        form.addRow('System-Stimme', self.system_cb)
        form.addRow('Tempo', speed_row)
        form.addRow('Lautstärke', self.volume_slider)
        form.addRow('', test_btn)
        form.addRow('', hint)
        form.addRow('', buttons)

        current = load_tts_settings()
        self._select(self.engine_cb, current.engine)
        self._select(self.piper_cb, current.piper_voice)
        self._select(self.system_cb, current.system_voice_id)
        self.speed_slider.setValue(int(round(current.speed * 100)))
        self.speed_label.setText(f"{self.speed_slider.value()} %")
        self.volume_slider.setValue(int(round(current.volume * 100)))

        self.engine_cb.currentIndexChanged.connect(self._update_visibility)
        self._update_visibility()

    @staticmethod
    def _select(combo, data):
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_visibility(self):
        piper = self.engine_cb.currentData() == tts_engine.ENGINE_PIPER
        self.piper_cb.setEnabled(piper)
        self.system_cb.setEnabled(not piper)

    def get_settings(self):
        return tts_engine.TTSSettings(
            engine=self.engine_cb.currentData(),
            piper_voice=self.piper_cb.currentData(),
            system_voice_id=self.system_cb.currentData() or '',
            speed=self.speed_slider.value() / 100.0,
            volume=self.volume_slider.value() / 100.0,
        )

    def on_test(self):
        self.speech_output.stop()
        self.speech_output.speak(prepare_for_speech(self.TEST_TEXT), settings=self.get_settings())


# (key, label) – the key is stored in the settings
PERSONALITY_MODES = [
    ('Default', 'Standard'),
    ('Friendly', 'Freundlich (du)'),
    ('Professional', 'Professionell (Sie)'),
    ('Custom', 'Eigener Prompt'),
    ('Off', 'Aus'),
]


def _personality_text_from_mode(mode: str, custom_text: str) -> str:
    if not mode or mode == 'Off':
        return ''
    if mode == 'Default':
        return ('Du bist ein hilfsbereiter, persönlicher Assistent. Beantworte die Fragen des Nutzers '
                'genau und verständlich und nutze dabei den bereitgestellten Kontext.')
    if mode == 'Friendly':
        return ('Du bist ein freundlicher, lockerer Assistent und duzt den Nutzer. '
                'Antworte kurz, positiv und hilfsbereit.')
    if mode == 'Professional':
        return ('Du bist ein professioneller Assistent und siezt den Nutzer. '
                'Antworte sachlich, höflich, präzise und knapp.')
    if mode == 'Custom':
        return custom_text or ''
    return ''


def list_ollama_models():
    """Names of the models installed in Ollama (empty if Ollama is not reachable)."""
    try:
        import ollama
        response = ollama.list()
        models = getattr(response, 'models', None)
        if models is None:
            models = response.get('models', [])
        names = []
        for m in models:
            name = getattr(m, 'model', None) or (m.get('model') or m.get('name') if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return sorted(names)
    except Exception as e:
        print(f"Could not list Ollama models: {e}")
        return []


class AssistantSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Assistent')
        self.setModal(True)

        form = QFormLayout(self)
        self.form = form

        self.model_cb = QComboBox()
        self.model_cb.setEditable(True)
        self.model_cb.addItems(list_ollama_models())
        self.model_cb.setToolTip('Ollama-Modell. Für bestes Deutsch ein Modell mit starker Deutsch-Unterstützung wählen.')

        self.language_cb = QComboBox()
        for key, label in ANSWER_LANGUAGES:
            self.language_cb.addItem(label, key)

        self.mode_cb = QComboBox()
        for key, label in PERSONALITY_MODES:
            self.mode_cb.addItem(label, key)
        self.custom_edit = QTextEdit()
        self.custom_edit.setPlaceholderText('Eigenen System-Prompt hier eingeben …')

        self.use_docs_cb = QCheckBox('Dokumente einbeziehen (RAG)')
        self.use_visual_cb = QCheckBox('Kamerabild einbeziehen')

        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(0, 30)
        self.memory_spin.setSuffix(' Wechsel')
        self.memory_spin.setToolTip('Wie viele vorherige Frage/Antwort-Paare der Assistent kennt (0 = kein Gedächtnis)')
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4096)
        self.max_tokens_spin.setSingleStep(50)
        self.max_tokens_spin.setSuffix(' Tokens')

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Abbrechen')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form.addRow('Sprachmodell', self.model_cb)
        form.addRow('Antwortsprache', self.language_cb)
        form.addRow('Persönlichkeit', self.mode_cb)
        form.addRow('Eigener Prompt', self.custom_edit)
        form.addRow('', self.use_docs_cb)
        form.addRow('', self.use_visual_cb)
        form.addRow('Gesprächsgedächtnis', self.memory_spin)
        form.addRow('Max. Antwortlänge', self.max_tokens_spin)
        form.addRow('', self.buttons)

        # Load settings
        settings = QSettings('YourOrg', 'SpeechToText')
        self.model_cb.setCurrentText(settings.value('assistant/model', MODEL_NAME))
        idx = self.language_cb.findData(settings.value('assistant/answer_language', 'de'))
        self.language_cb.setCurrentIndex(max(0, idx))
        idx = self.mode_cb.findData(settings.value('assistant/personality_mode', 'Default'))
        self.mode_cb.setCurrentIndex(max(0, idx))
        self.custom_edit.setPlainText(settings.value('assistant/custom_personality', ''))
        self.use_docs_cb.setChecked(settings.value('assistant/use_docs', True, type=bool))
        self.use_visual_cb.setChecked(settings.value('assistant/use_visual', True, type=bool))
        self.memory_spin.setValue(int(settings.value('assistant/memory_turns', DEFAULT_MEMORY_TURNS)))
        self.max_tokens_spin.setValue(int(settings.value('assistant/max_tokens', DEFAULT_MAX_TOKENS)))

        # Show/hide custom field
        def on_mode_change(_index=None):
            # hides the field together with its label
            self.form.setRowVisible(self.custom_edit, self.mode_cb.currentData() == 'Custom')
        on_mode_change()
        self.mode_cb.currentIndexChanged.connect(on_mode_change)

    def get_values(self):
        return {
            'model': self.model_cb.currentText().strip() or MODEL_NAME,
            'answer_language': self.language_cb.currentData(),
            'personality_mode': self.mode_cb.currentData(),
            'custom_personality': self.custom_edit.toPlainText(),
            'use_docs': bool(self.use_docs_cb.isChecked()),
            'use_visual': bool(self.use_visual_cb.isChecked()),
            'memory_turns': self.memory_spin.value(),
            'max_tokens': self.max_tokens_spin.value(),
        }


class SpeechSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Spracheingabe')
        self.setModal(True)
        form = QFormLayout(self)

        self.mic_cb = QComboBox()
        self.mic_cb.addItem('Standard-Mikrofon des Systems', '')
        try:
            for _idx, name in speech_input.list_input_devices():
                self.mic_cb.addItem(name, name)
        except Exception as e:
            self.mic_cb.addItem(f'(Geräte konnten nicht gelesen werden: {e})', '')

        self.model_cb = QComboBox()
        for key, label in speech_input.MODEL_SIZES:
            self.model_cb.addItem(label, key)
        self.language_cb = QComboBox()
        self.language_cb.addItems(list(speech_input.LANGUAGES))
        self.device_cb = QComboBox()
        self.device_cb.addItem('Automatisch (GPU, falls vorhanden)', 'auto')
        self.device_cb.addItem('GPU (CUDA)', 'cuda')
        self.device_cb.addItem('CPU', 'cpu')
        self.auto_stop_cb = QCheckBox('Aufnahme nach einer Sprechpause automatisch beenden')
        self.auto_send_cb = QCheckBox('Erkannten Text sofort senden')

        hint = QLabel('Für Deutsch liefert „large-v3-turbo“ mit Sprache „Deutsch“ die beste Erkennung. '
                      'Modelle werden beim ersten Benutzen einmalig heruntergeladen.')
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Abbrechen')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form.addRow('Mikrofon', self.mic_cb)
        form.addRow('Whisper-Modell', self.model_cb)
        form.addRow('Sprache', self.language_cb)
        form.addRow('Rechnen auf', self.device_cb)
        form.addRow('', self.auto_stop_cb)
        form.addRow('', self.auto_send_cb)
        form.addRow('', hint)
        form.addRow('', buttons)

        cfg = load_speech_settings()
        idx = self.mic_cb.findData(cfg['mic_name'])
        self.mic_cb.setCurrentIndex(idx if idx >= 0 else 0)
        idx = self.model_cb.findData(cfg['model_size'])
        if idx < 0:  # model not in the list (e.g. set manually): keep it selectable
            self.model_cb.addItem(cfg['model_size'], cfg['model_size'])
            idx = self.model_cb.count() - 1
        self.model_cb.setCurrentIndex(idx)
        self.language_cb.setCurrentText(cfg['language'])
        idx = self.device_cb.findData(cfg['compute_device'])
        self.device_cb.setCurrentIndex(idx if idx >= 0 else 0)
        self.auto_stop_cb.setChecked(cfg['auto_stop'])
        self.auto_send_cb.setChecked(cfg['auto_send'])

    def get_values(self):
        return {
            'mic_name': self.mic_cb.currentData() or '',
            'model_size': self.model_cb.currentData(),
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
        'language': _speech_language(settings.value('stt/language', speech_input.DEFAULT_LANGUAGE)),
        'compute_device': settings.value('stt/compute_device', 'auto'),
        'auto_stop': settings.value('stt/auto_stop', True, type=bool),
        'auto_send': settings.value('stt/auto_send', True, type=bool),
    }


def _speech_language(name):
    name = speech_input.LEGACY_LANGUAGE_NAMES.get(name, name)
    return name if name in speech_input.LANGUAGES else speech_input.DEFAULT_LANGUAGE


def save_speech_settings(values):
    settings = QSettings('YourOrg', 'SpeechToText')
    for key, value in values.items():
        settings.setValue(f'stt/{key}', value)


def _format_date(timestamp):
    """'2026-11-13 09:05:12.123' -> '13.11.2026 09:05'"""
    text = str(timestamp or "")
    try:
        date, time_ = text[:10], text[11:16]
        y, m, d = date.split("-")
        return f"{d}.{m}.{y} {time_}".strip()
    except ValueError:
        return text[:16]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KI-Assistent – Sprache, Kamera & Dokumente")
        self.setGeometry(100, 100, 1200, 800)

        # Database Init
        database_manager.init_db()
        
        # Session State
        self.session_id = str(uuid.uuid4())
        self.latest_visual_context = CAMERA_OFF_CONTEXT

        # Setup UI
        self.setup_ui()

        # Setup Workers
        settings = QSettings('YourOrg', 'SpeechToText')
        self.llm_worker = LLMWorker(
            settings.value('assistant/model', MODEL_NAME),
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

        # Speech output (TTS)
        self.speak_responses = True
        self.status_bridge = StatusBridge()
        self.status_bridge.message.connect(lambda msg: self.statusBar().showMessage(msg, 6000))
        self.speech_output = tts_engine.SpeechOutput(load_tts_settings(), on_status=self.status_bridge.message.emit)
        # Download/load the German voice in the background right away
        self.speech_output.preload()

        # Speech input (STT)
        self.speech_worker = None
        self._transcriber = None

        # Load assistant settings
        self.answer_language = settings.value('assistant/answer_language', 'de')
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
        file_menu = menubar.addMenu('Optionen')

        add_doc_action = QAction('Dokument hinzufügen …', self)
        add_doc_action.triggered.connect(self.add_document)
        file_menu.addAction(add_doc_action)
        
        new_session_action = QAction('Neues Gespräch', self)
        new_session_action.triggered.connect(self.start_new_session)
        file_menu.addAction(new_session_action)

        # Toggle whether assistant should speak responses
        speak_action = QAction('Antworten vorlesen', self, checkable=True)
        speak_action.setChecked(True)
        speak_action.toggled.connect(self.on_speak_responses_toggled)
        file_menu.addAction(speak_action)

        stop_speaking_action = QAction('Vorlesen abbrechen', self)
        stop_speaking_action.setShortcut(QKeySequence("Esc"))
        stop_speaking_action.triggered.connect(self.stop_speaking)
        file_menu.addAction(stop_speaking_action)
        file_menu.addSeparator()

        voice_settings_action = QAction('Sprachausgabe …', self)
        voice_settings_action.triggered.connect(self.open_voice_settings)
        file_menu.addAction(voice_settings_action)

        assistant_settings_action = QAction('Assistent …', self)
        assistant_settings_action.triggered.connect(self.open_assistant_settings)
        file_menu.addAction(assistant_settings_action)

        speech_settings_action = QAction('Spracheingabe …', self)
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
        new_session_btn = QPushButton("Neues Gespräch")
        new_session_btn.clicked.connect(self.start_new_session)
        self.session_list = QListWidget()
        self.session_list.setWordWrap(True)
        self.session_list.setStyleSheet(
            "QListWidget::item { padding: 6px 4px; border-bottom: 1px solid palette(midlight); }"
            "QListWidget::item:selected { background: palette(highlight); color: palette(highlighted-text); }")
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
        self.input_field.setPlaceholderText("Nachricht eingeben oder Strg+Leertaste zum Sprechen …")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Senden")
        self.send_btn.clicked.connect(self.send_message)

        self.mic_btn = QPushButton("Sprechen")
        self.mic_btn.setToolTip("Nachricht einsprechen (Strg+Leertaste)")
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
        
        self.camera_label = QLabel("Kamerabild")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.camera_label.setMinimumSize(400, 300)
        
        self.vision_status = QLabel("Erkannt: –")
        
        cam_controls = QHBoxLayout()
        self.start_cam_btn = QPushButton("Kamera starten")
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.stop_cam_btn = QPushButton("Kamera stoppen")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)

        self.yolo_mode_cb = QComboBox()
        self.yolo_mode_cb.addItem('Live-Erkennung', 'stream')
        self.yolo_mode_cb.addItem('Einzelbild', 'single')
        self.yolo_mode_cb.currentIndexChanged.connect(self.on_yolo_mode_changed)
        self.capture_btn = QPushButton("Bild analysieren")
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
        mode = self.yolo_mode_cb.currentData()
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
        self.capture_btn.setEnabled(self.yolo_mode_cb.currentData() == 'single')
        self.camera_label.setText("Kamera gestoppt")
        self.latest_visual_context = CAMERA_OFF_CONTEXT
        self.vision_status.setText("Erkannt: –")

    def on_camera_error(self, message):
        self.stop_camera()
        self.camera_label.setText(message)
        self.statusBar().showMessage(message, 5000)

    def on_yolo_mode_changed(self, _index=None):
        mode = self.yolo_mode_cb.currentData()
        if self.camera_worker:
            self.camera_worker.mode = mode
        self.capture_btn.setEnabled(mode == 'single')

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
                QMessageBox.warning(self, "Kamera", "Es konnte kein Bild von der Kamera aufgenommen werden.")
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
        self.vision_status.setText(f"Erkannt: {desc}")

    def open_voice_settings(self):
        dialog = VoiceSettingsDialog(self.speech_output, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            tts = dialog.get_settings()
            save_tts_settings(tts)
            self.speech_output.settings = tts
            self.speech_output.preload()
            voice = dialog.piper_cb.currentText() if tts.engine == tts_engine.ENGINE_PIPER else dialog.system_cb.currentText()
            self.statusBar().showMessage(f"Sprachausgabe: {voice}", 5000)

    def on_speak_responses_toggled(self, checked):
        self.speak_responses = checked
        if not checked:
            self.stop_speaking()

    def stop_speaking(self):
        self.speech_output.stop()

    def open_assistant_settings(self):
        dialog = AssistantSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.get_values()
            settings = QSettings('YourOrg', 'SpeechToText')
            for key, value in values.items():
                settings.setValue(f'assistant/{key}', value)
            # Applied to the next request without restart
            self.llm_worker.model_name = values['model']
            self.llm_worker.memory_turns = values['memory_turns']
            self.llm_worker.num_predict = values['max_tokens']
            self.answer_language = values['answer_language']
            self.assistant_personality_mode = values['personality_mode']
            self.assistant_custom_personality = values['custom_personality']
            self.use_docs = values['use_docs']
            self.use_visual = values['use_visual']
            self.personality_text = _personality_text_from_mode(values['personality_mode'],
                                                                values['custom_personality'])
            self.statusBar().showMessage(f"Assistent: {values['model']}", 5000)

    # ------------------------------------------------------------------ speech input

    def open_speech_settings(self):
        dialog = SpeechSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.get_values()
            save_speech_settings(values)
            self.statusBar().showMessage(
                f"Spracheingabe: Modell {values['model_size']}, Sprache {values['language']}", 5000)

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

        # Don't let the assistant talk over the user
        self.speech_output.stop()
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
            self.mic_btn.setText("Stopp")
            self.mic_level.setValue(0)
            self.mic_level.setVisible(True)
            self.statusBar().showMessage("Ich höre zu … (fertig? „Stopp“ oder Strg+Leertaste)")
        elif state == 'transcribing':
            self.mic_btn.setText("...")
            self.mic_btn.setEnabled(False)
            self.mic_level.setVisible(False)
            self.statusBar().showMessage("Erkenne Sprache …")

    def on_speech_text(self, text):
        self.statusBar().clearMessage()
        if not text:
            self.statusBar().showMessage("Keine Sprache erkannt.", 4000)
            return
        current = self.input_field.text().strip()
        self.input_field.setText(f"{current} {text}".strip())
        if self._speech_auto_send:
            self.send_message()
        else:
            self.input_field.setFocus()

    def on_speech_failed(self, message):
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Spracheingabe", f"Die Spracheingabe hat nicht funktioniert:\n{message}")

    def on_speech_finished(self):
        if self.speech_worker is not None:
            self.speech_worker.deleteLater()
        self.speech_worker = None
        self.mic_btn.setText("Sprechen")
        self.mic_btn.setEnabled(True)
        self.mic_level.setVisible(False)
        if self.pending_requests:
            self._update_thinking_status()

    # ------------------------------------------------------------------ documents

    def add_document(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Dokument auswählen", "",
                                                   "Dokumente (*.pdf *.txt *.md *.csv)")
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
            self.append_system_message(f"Fehler beim Kopieren von {filename}: {e}")
            return

        self.append_system_message(f"Füge {filename} hinzu … bitte warten.")
        worker = IngestWorker(dest_path)
        worker.ingest_finished.connect(self.on_ingest_finished)
        self._start_bg_worker(worker)

    def on_ingest_finished(self, filename, success):
        if success:
            self.append_system_message(f"{filename} wurde hinzugefügt.")
        else:
            self.append_system_message(f"{filename} konnte nicht hinzugefügt werden (Details in der Konsole).")

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
            title = (title or "(keine Nachrichten)").strip().replace("\n", " ")
            if len(title) > 60:
                title = title[:57] + "..."
            date = _format_date(start_time)
            item = QListWidgetItem(f"{title}\n{date}")
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            item.setToolTip(title)
            self.session_list.addItem(item)
            if session_id == self.session_id:
                item.setSelected(True)
                self.session_list.setCurrentItem(item)
        self.session_list.blockSignals(False)

    def _reset_chat_view(self):
        self.speech_output.stop()
        self._stream = None
        self.chat_display.clear()

    def start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self._reset_chat_view()
        self.append_system_message("Neues Gespräch gestartet.")
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
        delete_action = menu.addAction("Gespräch löschen")
        if menu.exec(self.session_list.mapToGlobal(pos)) == delete_action:
            session_id = item.data(Qt.ItemDataRole.UserRole)
            answer = QMessageBox.question(self, "Gespräch löschen",
                                          "Dieses Gespräch endgültig löschen?")
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
        self.chat_display.append(f"<b>Du:</b> {html.escape(text)}")
        self._scroll_chat_to_bottom()

    def append_assistant_message(self, text, error=False):
        if error:
            body = html.escape(text).replace("\n", "<br>")
            self.chat_display.append(f"<b>Assistent:</b> <span style='color:#c0392b'>{body}</span>")
        else:
            self.chat_display.append("<b>Assistent:</b>")
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
            self.statusBar().showMessage(f"Assistent denkt nach … ({self.pending_requests} offen)")
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
            use_visual=self.use_visual,
            answer_language=self.answer_language,
        )

    def _speak(self, text):
        text = prepare_for_speech(text)
        if self.speak_responses and text:
            self.speech_output.speak(text)

    def on_response_chunk(self, request_id, session_id, text):
        if session_id != self.session_id:
            return  # answer for a session that is not shown
        if self._stream is None or self._stream['rid'] != request_id:
            # First chunk: open a new "Assistant:" paragraph and keep a cursor to it.
            # The cursor stays inside this paragraph even if the user sends another
            # message meanwhile (which is appended below it).
            self.chat_display.append("<b>Assistent:</b>")
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
        self.speech_output.shutdown()
        for worker in list(self._bg_workers):
            worker.wait(2000)
        super().closeEvent(event)


def install_german_translations(app):
    """German texts for Qt's standard dialogs and buttons (Ja/Nein, Abbrechen, Öffnen, ...)."""
    from PyQt6.QtCore import QTranslator, QLibraryInfo, QLocale
    QLocale.setDefault(QLocale(QLocale.Language.German, QLocale.Country.Germany))
    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load("qtbase_de", path):
        app.installTranslator(translator)
    return translator


if __name__ == "__main__":
    app = QApplication(sys.argv)
    install_german_translations(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
