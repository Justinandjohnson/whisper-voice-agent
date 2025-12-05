import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import tempfile
import os
import sys
import ssl

# Bypass SSL verification for model download
ssl._create_default_https_context = ssl._create_unverified_context

def record_audio(duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return recording, fs

def main():
    print("Loading Whisper model (base)...")
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("\nReady to talk!")
    while True:
        try:
            input("Press Enter to start recording (5 seconds)...")
            
            recording, fs = record_audio()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name
                
            # Convert to int16 for wavfile write
            recording_int16 = (recording * 32767).astype(np.int16)
            wav.write(temp_filename, fs, recording_int16)
            
            print("Transcribing...")
            result = model.transcribe(temp_filename)
            print(f"\nYou said: {result['text']}\n")
            
            os.remove(temp_filename)
            
            cont = input("Record again? (y/n): ")
            if cont.lower() != 'y':
                break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
