# AI Assistant mit Vision & RAG

Lokaler KI-Assistent mit Desktop-GUI (PyQt6):

- **Chat** mit einem lokalen LLM über [Ollama](https://ollama.com)
- **Dokumente (RAG):** PDF, TXT, MD und CSV werden indexiert und als Kontext genutzt (Chroma + `bge-m3` Embeddings)
- **Vision:** Webcam mit YOLOv8-Objekterkennung; was die Kamera sieht, fließt in die Antwort ein
- **Sprachausgabe (TTS)** über `pyttsx3`, Stimme/Tempo/Lautstärke einstellbar
- **Chatverlauf** wird in einer lokalen SQLite-Datenbank (`chat_history.db`) gespeichert

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
   Das YOLO-Modell (`yolov8n.pt`) lädt `ultralytics` beim ersten Kamerastart automatisch herunter.

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
| `legacy/voice_assistant.py` | Alter Konsolen-Sprachassistent (llama-cpp + Google-Spracherkennung). Wird durch die Spracheingabe in der GUI ersetzt. |

## Bedienung

- **Options → Add Document:** Dokument hinzufügen. Es wird nach `documents/` kopiert und im Hintergrund indexiert. Erneutes Hinzufügen derselben Datei ersetzt den alten Stand.
- Dateien, die direkt in den Ordner `documents/` gelegt werden, werden beim nächsten Start automatisch indexiert. Zum kompletten Neuaufbau den Ordner `chrome_langchain_db/` löschen.
- **Options → Assistant Settings:** Persönlichkeit/System-Prompt, Dokumente und Kamera-Kontext an- und abschalten.
- **Options → Voice Settings / Speak Responses:** Sprachausgabe konfigurieren.
- **Kamera:** Modus „Stream“ (laufende Erkennung, gedrosselt) oder „Single“ (Erkennung nur per „Capture Frame“).
- Nachrichten können auch geschickt werden, während das Modell noch lädt oder antwortet. Sie werden der Reihe nach abgearbeitet. Das LLM wird beim Start vorgeladen, um die erste Antwort zu beschleunigen.

## Lokale Daten

`models/`, `documents/`, `chrome_langchain_db/`, `chat_history.db` und `*.pt` sind in `.gitignore` eingetragen und werden nicht eingecheckt.
