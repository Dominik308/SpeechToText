# Voice Assistant Setup

## Prerequisites
- Windows 10/11
- NVIDIA GPU (RTX 3060Ti or similar) with drivers installed
- Python 3.10 or 3.11 recommended

## Installation

1. **Install dependencies:**
   It is recommended to install `llama-cpp-python` separately first to ensure CUDA (GPU) support.
   
   If you have CUDA 12.x installed:
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
   ```
   (Or check https://github.com/abetlen/llama-cpp-python for the wheel matching your CUDA version).

   Then install the rest:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the LLM:**
   Run the download script to fetch the Llama-3 model (approx 4.6GB).
   ```bash
   python download_model.py
   ```

## Usage

1. **Audio Setup:**
   - Connect your Bose NC 700 HP.
   - Set it as the **Default Playback Device** in Windows Sound settings (for TTS output).
   - The script will automatically try to find it for **Microphone Input**.

2. **Run the Assistant:**
   ```bash
   python voice_assistant.py
   ```

## Controls
- Speak clearly into the microphone.
- Say "stop system" to exit.
- The assistant will see what is on "Cam One" and include it in its response.
- YOLO Mode: You can choose "Stream" (continuous inference with throttling and downscale) or "Single" (capture one frame for detection using the "Capture Frame" button).
- The LLM is pre-warmed on startup to reduce the first-query delay. TTS uses a persistent engine to avoid repeated initialization overhead.
