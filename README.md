# AI Assistant mit Vision & RAG

Lokaler KI-Assistent mit Desktop-GUI (PyQt6):

- **Chat** mit einem lokalen LLM über [Ollama](https://ollama.com): Antworten erscheinen Wort für Wort,
  der Assistent erinnert sich an den bisherigen Gesprächsverlauf
- **Spracheingabe** per Mikrofon, lokal erkannt mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (keine Cloud)
- **Dokumente (RAG):** PDF, TXT, MD und CSV werden indexiert und als Kontext genutzt (Chroma + `bge-m3` Embeddings)
- **Vision:** Webcam mit YOLOv8-Objekterkennung; was die Kamera sieht, fließt in die Antwort ein
- **Sprachausgabe (TTS)** über `pyttsx3`, Satz für Satz schon während die Antwort entsteht; Stimme/Tempo/Lautstärke einstellbar
- **Sitzungen:** Alle Gespräche werden lokal gespeichert (`chat_history.db`) und lassen sich links in der Liste wieder öffnen und fortsetzen

## Voraussetzungen

- Windows 10/11 (Linux funktioniert grundsätzlich ebenfalls)
- Python 3.10 oder 3.11
- NVIDIA-GPU empfohlen (z. B. RTX 3060 Ti)
- [Ollama](https://ollama.com/download) installiert und gestartet

## Installation

1. **Python-Pakete installieren:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Modelle in Ollama laden:**
   ```bash
   ollama pull mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso
   ollama pull bge-m3
   ```
   Das YOLO-Modell (`yolov8n.pt`) lädt `ultralytics` beim ersten Kamerastart automatisch herunter,
   das Whisper-Modell für die Spracheingabe wird bei der ersten Aufnahme von Hugging Face geladen.

3. **Optional: Spracheingabe auf der GPU.** faster-whisper braucht dafür CUDA 12 mit cuBLAS und cuDNN 9.
   - **Windows:** CUDA Toolkit 12 und cuDNN 9 von NVIDIA installieren (die `bin`-Ordner müssen im `PATH` liegen),
     alternativ die fertigen DLLs aus [Purfview/whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases)
     in einen Ordner im `PATH` legen.
   - **Linux:** `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`

   Fehlen die Bibliotheken, läuft die Erkennung automatisch auf der CPU (langsamer, aber funktionsfähig).

## Starten

```bash
python gui_app.py
```

Weitere Einstiegspunkte:

| Datei | Zweck |
|---|---|
| `gui_app.py` | Haupt-Anwendung (GUI) |
| `main.py` | Einfacher Konsolen-Chat mit den Dokumenten (RAG) |
| `check_audio_devices.py` | Listet Audio-Geräte (hilfreich für die Mikrofon-Auswahl) |
| `speech_input.py` | Mikrofon-Aufnahme und Spracherkennung (von der GUI genutzt) |

## Bedienung

- **Spracheingabe:** Button „Speak“ oder **Strg+Leertaste**. Nach einer kurzen Sprechpause stoppt die Aufnahme automatisch
  (oder erneut „Stop“ / Strg+Leertaste drücken), der Text wird erkannt und direkt gesendet.
- **Options → Speech Input Settings:** Mikrofon, Whisper-Modell (`tiny` … `large-v3-turbo`), Sprache (Auto/Deutsch/Englisch),
  GPU/CPU, automatisches Stoppen und automatisches Senden. Standard ist das Modell `small`; für bessere deutsche
  Erkennung auf der GPU empfiehlt sich `large-v3-turbo`.
- **Sitzungen:** „New Session“ startet ein neues Gespräch, ein Klick auf einen Eintrag links lädt ein altes Gespräch
  und führt es fort. Rechtsklick → „Delete Session“ löscht es.
- **Options → Assistant Settings:** Persönlichkeit/System-Prompt, Dokumente und Kamera-Kontext an- und abschalten,
  **Gesprächsgedächtnis** (wie viele vorherige Frage/Antwort-Paare das Modell sieht, Standard 6, 0 = aus) und
  **maximale Antwortlänge** (Standard 400 Tokens).
- **Options → Add Document:** Dokument hinzufügen. Es wird nach `documents/` kopiert und im Hintergrund indexiert. Erneutes Hinzufügen derselben Datei ersetzt den alten Stand.
- Dateien, die direkt in den Ordner `documents/` gelegt werden, werden beim nächsten Start automatisch indexiert. Zum kompletten Neuaufbau den Ordner `chrome_langchain_db/` löschen.
- **Options → Voice Settings / Speak Responses:** Sprachausgabe konfigurieren. Beim Start einer Spracheingabe wird noch
  ausstehende Sprachausgabe verworfen, damit der Assistent nicht dazwischenredet.
- **Kamera:** Modus „Stream“ (laufende Erkennung, gedrosselt) oder „Single“ (Erkennung nur per „Capture Frame“).
- Nachrichten können auch geschickt werden, während das Modell noch lädt oder antwortet. Sie werden der Reihe nach abgearbeitet. Das LLM wird beim Start vorgeladen, um die erste Antwort zu beschleunigen.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Lokale Daten

`models/`, `documents/`, `chrome_langchain_db/`, `chat_history.db` und `*.pt` sind in `.gitignore` eingetragen und werden nicht eingecheckt.
