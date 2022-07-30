import os
import torchaudio
from recognizer import SpeechRecognizer, VietnameseSpeechRecognizer
from synthesizer import SpeechSynthesizer, EnglishSpeechSynthesizer
from translator import Translator, LanguageTranslator


class AudioTranslator(object):
    def __init__(self, mode="vi2en"):
        self.recognizer = VietnameseSpeechRecognizer()
        self.synthesizer = EnglishSpeechSynthesizer()
        self.translator = LanguageTranslator() 

    def __call__(self, x):
        x = self.recognizer(x)
        x = self.translator(x)
        x, r = self.synthesizer(x)

        return x, r


if __name__ == "__main__":
    audio_translator = AudioTranslator()
    x, r = audio_translator("data/001.wav")
    torchaudio.save("data/ex_out_001.wav", x, r)

