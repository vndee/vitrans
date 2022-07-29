import os
import wave
import pyaudio
import sounddevice
import torchaudio
import numpy as np
from pynput import keyboard
from playsound import playsound
from threading import Thread, Lock
from scipy.io.wavfile import write as wavwrite

from recognizer import VietnameseSpeechRecognizer
from synthesizer import EnglishSpeechSynthesizer
from translator import LanguageTranslator


TMP_DIR = "./tmp/"
SOURCE_AUDIO_TMP_FILE = os.path.join(TMP_DIR, "source.wav")
TARGET_AUDIO_TMP_FILE = os.path.join(TMP_DIR, "target.wav")


class Listener(keyboard.Listener):
    def __init__(self, recorder, player, recognizer, synthesizer, translator):
        super().__init__(on_press = self.on_press, on_release = self.on_release)
        self.text = None
        self.recorder = recorder
        self.player = player
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.translator = translator
    
    def on_press(self, key):
        if key is None: # unknown event
            pass
        elif isinstance(key, keyboard.Key): # special key event
            if key.ctrl and self.player.playing == 0:
                self.recorder.start()
        elif isinstance(key, keyboard.KeyCode): # alphanumeric key event
            if key.char == 'q': # press q to quit
                if self.recorder.recording:
                    self.recorder.stop()
                return False # this is how you stop the listener thread
            if key.char == 'p' and not self.recorder.recording:
                translated_text = self.translator(self.text)
                print("Output:", translated_text)
                wav, rate = self.synthesizer(translated_text)
                sounddevice.play(wav.detach().numpy().T, samplerate=rate)
                sounddevice.wait()
                torchaudio.save(TARGET_AUDIO_TMP_FILE, wav, rate)
                
    def on_release(self, key):
        if key is None: # unknown event
            pass
        elif isinstance(key, keyboard.Key): # special key event
            if key.ctrl:
                self.recorder.stop()
                self.text = self.recognizer(SOURCE_AUDIO_TMP_FILE)
                print("Input:", self.text)
        elif isinstance(key, keyboard.KeyCode): # alphanumeric key event
            pass


class Recorder:
    def __init__(self, 
                 wavfile, 
                 chunksize=8192, 
                 dataformat=pyaudio.paInt16, 
                 channels=1, 
                 rate=16000):
        self.filename = wavfile
        self.chunksize = chunksize
        self.dataformat = dataformat
        self.channels = channels
        self.rate = rate
        self.recording = False
        self.pa = pyaudio.PyAudio()

    def start(self):
        #we call start and stop from the keyboard listener, so we use the asynchronous 
        # version of pyaudio streaming. The keyboard listener must regain control to 
        # begin listening again for the key release.
        if not self.recording:
            self.wf = wave.open(self.filename, 'wb')
            self.wf.setnchannels(self.channels)
            self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
            self.wf.setframerate(self.rate)
            
            def callback(in_data, frame_count, time_info, status):
                #file write should be able to keep up with audio data stream (about 1378 Kbps)
                self.wf.writeframes(in_data) 
                return (in_data, pyaudio.paContinue)
            
            self.stream = self.pa.open(format = self.dataformat,
                                       channels = self.channels,
                                       rate = self.rate,
                                       input = True,
                                       stream_callback = callback)
            self.stream.start_stream()
            self.recording = True
            print('recording started')
    
    def stop(self):
        if self.recording:         
            self.stream.stop_stream()
            self.stream.close()
            self.wf.close()
            
            self.recording = False
            print('recording finished')


class Player:
    def __init__(self, wavfile):
        self.wavfile = wavfile
        self.playing = 0 # flag so we don't try to record while the wav file is in use
        self.lock = Lock() # mutex so incrementing and decrementing self.playing is safe
    
    # contents of the run function are processed in another thread so we use the blocking
    # version of pyaudio play file example: http://people.csail.mit.edu/hubert/pyaudio/#play-wave-example
    def run(self):
        print("playing recorded audio")

        with self.lock:
            self.playing += 1
        with wave.open(self.wavfile, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(8192)
            while data != b'':
                stream.write(data)
                data = wf.readframes(8192)

            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
        with self.lock:
            self.playing -= 1
        
    def start(self):
        Thread(target=self.run).start()


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    r = Recorder(SOURCE_AUDIO_TMP_FILE)
    p = Player(TARGET_AUDIO_TMP_FILE)

    recognizer = VietnameseSpeechRecognizer()
    synthesizer = EnglishSpeechSynthesizer()
    translator = LanguageTranslator()

    l = Listener(r, p, recognizer, synthesizer, translator)
    print('hold ctrl to record, press p to playback, press q to quit')
    l.start() # keyboard listener is a thread so we start it here
    l.join() # wait for the tread to terminate so the program doesn't instantly close
 
