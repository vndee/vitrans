import gc
import torch
import soundfile as sf

from typing import Optional
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class SpeechRecognizer(object):
    def __call__(self, x: str) -> Optional[str]:
        pass


class VietnameseSpeechRecognizer(SpeechRecognizer):
    def __init__(self):
        self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
        self.ctc_decoder: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

    def __call__(self, x: str):
        speech, _ = sf.read(x)
        input_values = self.processor(speech, return_tensors="pt", padding="longest").input_values
        logits = self.ctc_decoder(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        gc.collect()
        return transcription[0]


#if __name__ == "__main__":
#    worker = VietnameseSpeechRecognizer()
#    print(worker("data/001.wav"))
