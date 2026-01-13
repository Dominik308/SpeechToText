import pyaudio

p = pyaudio.PyAudio()
print("Audio Devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    try:
        name = info.get('name')
        inputs = info.get('maxInputChannels')
        print(f"Index {i}: {name} (Inputs: {inputs})")
    except Exception as e:
        print(f"Index {i}: Error reading info - {e}")

p.terminate()
