# KI-Assistent mit Sprache, Kamera & Dokumenten

Lokaler KI-Assistent mit Desktop-GUI (PyQt6), **auf Deutsch optimiert**: Oberfläche, Spracherkennung,
Antworten und Sprachausgabe sind standardmäßig auf beste deutsche Qualität eingestellt.
Alles läuft lokal, nichts geht in die Cloud.

- **Chat** mit einem lokalen LLM über [Ollama](https://ollama.com): Antworten erscheinen Wort für Wort,
  der Assistent erinnert sich an den bisherigen Gesprächsverlauf
- **Spracheingabe** per Mikrofon, lokal erkannt mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (Standard: Modell `large-v3-turbo`, Sprache Deutsch)
- **Dokumente (RAG):** PDF, TXT, MD und CSV werden indexiert und als Kontext genutzt (Chroma + `bge-m3` Embeddings)
- **Vision:** Webcam mit YOLOv8-Objekterkennung; was die Kamera sieht, fließt in die Antwort ein
- **Natürliche deutsche Sprachausgabe** mit [Piper](https://github.com/OHF-Voice/piper1-gpl) (Standard: Stimme „Thorsten“, hohe Qualität),
  Satz für Satz schon während die Antwort entsteht; Rückfall auf die Windows-Stimme, falls Piper nicht verfügbar ist
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
   Das YOLO-Modell (`yolov8n.pt`) lädt `ultralytics` beim ersten Kamerastart automatisch herunter.
   Das Whisper-Modell (Spracheingabe, ca. 1,6 GB) und die deutsche Piper-Stimme (Sprachausgabe) werden beim ersten
   Benutzen automatisch von Hugging Face geladen und danach lokal verwendet.

3. **Optional: Spracheingabe auf der GPU.** faster-whisper braucht dafür CUDA 12 mit cuBLAS und cuDNN 9.
   - **Windows:** CUDA Toolkit 12 und cuDNN 9 von NVIDIA installieren (die `bin`-Ordner müssen im `PATH` liegen),
     alternativ die fertigen DLLs aus [Purfview/whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases)
     in einen Ordner im `PATH` legen.
   - **Linux:** `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`

   Fehlen die Bibliotheken, läuft die Erkennung automatisch auf der CPU (langsamer, aber funktionsfähig).

## Deutsch: was wofür sorgt

| Baustein | Einstellung für bestes Deutsch | Wo änderbar |
|---|---|---|
| Spracherkennung | Whisper `large-v3-turbo`, Sprache fest auf Deutsch, deutscher Startprompt für Rechtschreibung und Zeichensetzung, Filter für typische Whisper-Phantomsätze („Untertitel im Auftrag des ZDF …“) | Optionen → Spracheingabe |
| Sprachmodell | Deutscher System-Prompt, Antwortsprache „Immer Deutsch“, Kamera-Objekte auf Deutsch („2× Person, 1× Tasse“) | Optionen → Assistent |
| Sprachausgabe | Piper-Stimme „Thorsten (hohe Qualität)“; Abkürzungen, Einheiten und Datumsangaben werden ausgeschrieben („ca. 21 °C“ → „circa 21 Grad Celsius“, „am 3. Oktober“ → „am dritten Oktober“) | Optionen → Sprachausgabe |
| Dokumentensuche | Embedding-Modell `bge-m3` ist mehrsprachig und funktioniert gut mit deutschen Texten | – |

**Sprachmodell wählen:** Die Antwortqualität auf Deutsch hängt am stärksten vom Ollama-Modell ab. Unter
Optionen → Assistent lässt sich jedes installierte Modell auswählen. Es lohnt sich, das voreingestellte Llama-3.1-8B-Modell
mit Modellen mit starker Deutsch-Unterstützung zu vergleichen (z. B. aktuelle Qwen- oder Gemma-Modelle oder deutsche
Feinabstimmungen wie *SauerkrautLM*). Auf einer 8-GB-Grafikkarte (RTX 3060 Ti) passen Modelle bis etwa 8 Milliarden
Parameter (4-Bit) gut neben Whisper; größere Modelle werden langsamer.

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
| `tts_engine.py` | Sprachausgabe mit Piper bzw. System-Stimme (von der GUI genutzt) |

## Bedienung

- **Spracheingabe:** Button „Sprechen“ oder **Strg+Leertaste**. Nach einer kurzen Sprechpause stoppt die Aufnahme automatisch
  (oder „Stopp“ / Strg+Leertaste drücken), der Text wird erkannt und direkt gesendet.
- **Vorlesen abbrechen:** **Esc**. Startet man eine Spracheingabe, verstummt der Assistent ebenfalls sofort.
- **Optionen → Spracheingabe:** Mikrofon, Whisper-Modell, Sprache (Deutsch/Automatisch/Englisch), GPU/CPU,
  automatisches Stoppen und automatisches Senden.
- **Optionen → Sprachausgabe:** Piper-Stimme (Thorsten, Kerstin, Ramona, Eva …) oder Windows-Stimme, Tempo, Lautstärke,
  „Stimme testen“.
- **Gespräche:** „Neues Gespräch“ startet ein neues Gespräch, ein Klick auf einen Eintrag links lädt ein altes Gespräch
  und führt es fort. Rechtsklick → „Gespräch löschen“ löscht es.
- **Optionen → Assistent:** Sprachmodell, Antwortsprache, Persönlichkeit (Standard, Freundlich/du, Professionell/Sie,
  eigener Prompt), Dokumente und Kamerabild an- und abschalten, **Gesprächsgedächtnis** (Standard 6 Wechsel, 0 = aus)
  und **maximale Antwortlänge** (Standard 400 Tokens).
- **Optionen → Dokument hinzufügen:** Dokument hinzufügen. Es wird nach `documents/` kopiert und im Hintergrund indexiert. Erneutes Hinzufügen derselben Datei ersetzt den alten Stand.
- Dateien, die direkt in den Ordner `documents/` gelegt werden, werden beim nächsten Start automatisch indexiert. Zum kompletten Neuaufbau den Ordner `chrome_langchain_db/` löschen.
- **Kamera:** Modus „Live-Erkennung“ (laufend, gedrosselt) oder „Einzelbild“ (Erkennung nur per „Bild analysieren“).
- Nachrichten können auch geschickt werden, während das Modell noch lädt oder antwortet. Sie werden der Reihe nach abgearbeitet. Das LLM wird beim Start vorgeladen, um die erste Antwort zu beschleunigen.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Lokale Daten

`models/`, `documents/`, `chrome_langchain_db/`, `chat_history.db` und `*.pt` sind in `.gitignore` eingetragen und werden nicht eingecheckt.
