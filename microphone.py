import pyaudio
from pyaudio import PyAudio


class Microphone(object):
    def __init__(self, device_id: int = None, sample_rate: int = None, buffer_size: int = 1024):
        super(Microphone, self).__init__()
        self.device_id = device_id
        self.SAMPLE_RATE = sample_rate
        self.BUFFER_SIZE = buffer_size
        self.SAMPLE_WIDTH = pyaudio.get_sample_size(pyaudio.paInt16)

        self.audio = None
        self.stream = None

    def __enter__(self):
        self.audio = PyAudio()

        try:
            self.stream = Microphone.Stream(
                self.audio.open(
                    input_device_index=self.device_id,
                    channels=1,
                    format=pyaudio.paInt16,
                    rate=self.SAMPLE_RATE,
                    frames_per_buffer=self.BUFFER_SIZE,
                    input=True
                )
            )
        except Exception:
            self.audio.terminate()

        return self

    def __exit__(self, exec_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()


    class Stream(object):
        def __init__(self, pyaudio_stream):
            self.pyaudio_stream = pyaudio_stream

        def read(self, size):
            return self.pyaudio_stream.read(size, exception_on_overflow=False)

        def close(self):
            try:
                if not self.pyaudio_stream.is_stopped():
                    self.pyaudio_stream.stop_stream()
            finally:
                self.pyaudio_stream.close()

