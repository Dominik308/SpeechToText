import speech_recognition as sr
import pyttsx3
import cv2
import threading
import time
from ultralytics import YOLO
try:
    from llama_cpp import Llama
except ImportError:
    print("Please install llama-cpp-python for LLM support: pip install llama-cpp-python")
    Llama = None

# Configuration
CAMERA_INDEX = 0  # 0 is usually the default webcam
AUDIO_DEVICE_NAME_SUBSTRING = "Bose NC 700 HP"
MODEL_PATH_LLM = "models/blobs/sha256-7b4601d271b2988c5150ba081dfd9263197b5b7f2d39b25e8bad601380019031" # Updated model path from checking manifest
MODEL_PATH_YOLO = "yolov8n.pt" # Nano model for speed

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone_index = self.find_audio_device_index(AUDIO_DEVICE_NAME_SUBSTRING)
        
        # Initialize Audio Engine
        self.engine = pyttsx3.init()
        # Note: pyttsx3 usually uses the system default output. 
        # To output strictly to Bose, ensure it is the default playback device in Windows.
        
        # Initialize Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Initialize AI Models
        print("Loading YOLO model...")
        self.vision_model = YOLO(MODEL_PATH_YOLO)
        
        print("Loading LLM model (this may take time)...")
        if Llama:
            # n_gpu_layers=-1 attempts to offload all layers to GPU (RTX 3060Ti)
            # n_ctx=8192 matches the model's training context to avoid warnings
            self.llm = Llama(
                model_path=MODEL_PATH_LLM,
                n_ctx=8192,
                n_gpu_layers=-1, 
                verbose=True  # Set to True to see loading details if it gets stuck
            )
        else:
            self.llm = None
            print("WARNING: LLM not initialized. Chat will be limited.")

    def find_audio_device_index(self, name_substring):
        import pyaudio
        p = pyaudio.PyAudio()
        print(f"Searching for audio device containing '{name_substring}'...")
        
        candidate = None
        
        # Iterate all devices
        for i in range(p.get_device_count()):
            try:
                dev = p.get_device_info_by_index(i)
                name = dev.get('name', '')
                inputs = dev.get('maxInputChannels', 0)
                
                if name_substring.lower() in name.lower() and inputs > 0:
                    # Priority: Hands-Free
                    if "hands-free" in name.lower() or "headset" in name.lower():
                        print(f"Found Priority Device: {name} at index {i} (Inputs: {inputs})")
                        p.terminate()
                        return i
                    
                    # Store as candidate if we don't find a priority one
                    if candidate is None:
                        candidate = (i, name)
            except Exception as e:
                pass
        
        p.terminate()
        
        if candidate:
            print(f"Found Device: {candidate[1]} at index {candidate[0]}")
            return candidate[0]
            
        print(f"Device '{name_substring}' not found. Using default.")
        return None

    def speak(self, text):
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def get_visual_context(self):
        if not self.cap.isOpened():
            return "Camera not available."
        
        ret, frame = self.cap.read()
        if not ret:
            return "Failed to capture image."
        
        # Run YOLO inference
        results = self.vision_model(frame, verbose=False)
        
        # Extract labels
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.vision_model.names[class_id]
                detected_objects.append(label)
        
        if not detected_objects:
            return "empty scene"
        
        # Unique counts
        from collections import Counter
        counts = Counter(detected_objects)
        description = ", ".join([f"{count} {obj}" for obj, count in counts.items()])
        return description

    def generate_response(self, user_text, visual_context):
        if not self.llm:
            return "I heard you, but my LLM brain is not connected."
            
        prompt = f"""<|system|>
You live in a computer with eyes.
Visual Context: The camera sees {visual_context}.
User Input: {user_text}
Answer briefly and helpfully.
<|user|>
{user_text}
<|assistant|>
"""
        output = self.llm(
            prompt, 
            max_tokens=150, 
            stop=["<|user|>", "\n"], 
            echo=False
        )
        return output['choices'][0]['text'].strip()

    def process_interaction(self, text):
        """Common logic for processing text from any source (voice or keyboard)"""
        if "stop system" in text.lower():
            self.speak("Shutting down.")
            return False
            
        # 1. Get Visual Context
        scene_desc = self.get_visual_context()
        print(f"[Vision]: Saw {scene_desc}")
        
        # 2. Get LLM Response
        response = self.generate_response(text, scene_desc)
        
        # 3. Speak
        self.speak(response)
        return True

    def run_keyboard(self):
        print("\n--- Keyboard Mode Active ---")
        self.speak("Keyboard mode activated. Please type your message.")
        
        while True:
            try:
                text = input("\nYou: ").strip()
                if not text: continue
                
                # Check for exit command
                if not self.process_interaction(text):
                    break
                    
            except KeyboardInterrupt:
                break
        
        self.cleanup()

    def run_voice(self):
        self.speak("Voice mode activated. Listening...")
        print(f"DEBUG: Using Microphone Index: {self.microphone_index}")
        
        try:
            # Try-catch block specifically for microphone initialization
            try:
                source = sr.Microphone(device_index=self.microphone_index, sample_rate=16000)
                # Test opening the stream briefly to ensure it works
                with source:
                    pass
            except Exception as e:
                print(f"Failed to use specific device index {self.microphone_index}: {e}")
                print("Falling back to System Default Microphone...")
                self.microphone_index = None
                source = sr.Microphone(device_index=None)

            with source:
                # Skipping adjust_for_ambient_noise which often crashes with Bluetooth headsets on Windows
                # We set a sensible default energy_threshold instead
                self.recognizer.energy_threshold = 1000 
                self.recognizer.dynamic_energy_threshold = False # Disable dynamic adjustment to prevent crashes
                print(f"Ready. Listening on device index: {self.microphone_index if self.microphone_index is not None else 'Default'}")
                
                while True:
                    try:
                        print("Listening...")
                        audio = self.recognizer.listen(source)
                        print("Processing...")
                        
                        try:
                            # Standard recognition
                            text = self.recognizer.recognize_google(audio)
                            print(f"User: {text}")
                            
                            if not self.process_interaction(text):
                                break
                            
                        except sr.UnknownValueError:
                            print("Audio not understood. (Silence or low confidence)")
                            # Optional: Try to get raw list to see partial matches
                            try:
                                raw_json = self.recognizer.recognize_google(audio, show_all=True)
                                if raw_json:
                                    print(f"DEBUG - Partial/Low confidence results: {raw_json}")
                            except:
                                pass
                        except sr.RequestError as e:
                            print(f"Could not request results; {e}")
                            
                    except KeyboardInterrupt:
                        break
        except Exception as e:
            print(f"Error accessing microphone: {e}")
            
        self.cleanup()

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        print("\n=== Voice & Vision Assistant ===")
        print("Select Input Mode:")
        print("1. Voice (Microphone)")
        print("2. Keyboard (Text)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            self.run_keyboard()
        else:
            self.run_voice()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
