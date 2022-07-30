import gc
import re
import torch
import torchaudio
import sounddevice
import unicodedata

import numpy as np

from argparse import Namespace
from pathlib import Path
from typing import Optional
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from t2s.hifigan.mel2wave import mel2wave
from t2s.nat.text2mel import text2mel


class FLAGS(Namespace):
    """Configurations"""

    duration_lstm_dim = 256
    vocab_size = 256
    duration_embed_dropout_rate = 0.5
    num_training_steps = 200_000
    postnet_dim = 512
    acoustic_decoder_dim = 512
    acoustic_encoder_dim = 256

    # dataset
    max_phoneme_seq_len = 256 * 1
    assert max_phoneme_seq_len % 256 == 0  # prevent compilation error on Colab T4 GPU
    max_wave_len = 1024 * 64 * 3

    # Montreal Forced Aligner
    special_phonemes = ["sil", "sp", "spn", " "]  # [sil], [sp] [spn] [word end]
    sil_index = special_phonemes.index("sil")
    sp_index = sil_index  # no use of "sp"
    word_end_index = special_phonemes.index(" ")
    _normal_phonemes = (
        []
        + ["a", "b", "c", "d", "e", "g", "h", "i", "k", "l"]
        + ["m", "n", "o", "p", "q", "r", "s", "t", "u", "v"]
        + ["x", "y", "à", "á", "â", "ã", "è", "é", "ê", "ì"]
        + ["í", "ò", "ó", "ô", "õ", "ù", "ú", "ý", "ă", "đ"]
        + ["ĩ", "ũ", "ơ", "ư", "ạ", "ả", "ấ", "ầ", "ẩ", "ẫ"]
        + ["ậ", "ắ", "ằ", "ẳ", "ẵ", "ặ", "ẹ", "ẻ", "ẽ", "ế"]
        + ["ề", "ể", "ễ", "ệ", "ỉ", "ị", "ọ", "ỏ", "ố", "ồ"]
        + ["ổ", "ỗ", "ộ", "ớ", "ờ", "ở", "ỡ", "ợ", "ụ", "ủ"]
        + ["ứ", "ừ", "ử", "ữ", "ự", "ỳ", "ỵ", "ỷ", "ỹ"]
    )

    # dsp
    mel_dim = 80
    n_fft = 1024
    sample_rate = 16000
    fmin = 0.0
    fmax = 8000

    # training
    batch_size = 64
    learning_rate = 1e-4
    duration_learning_rate = 1e-4
    max_grad_norm = 1.0
    weight_decay = 1e-4
    token_mask_prob = 0.1

    # ckpt
    ckpt_dir = Path("resources/")
    data_dir = Path("train_data")


class SpeechSynthesizer(object):
    def __call__(self, x) -> (Optional[torch.Tensor], Optional[int]):
        pass


class EnglishSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self):
        self.models, self.cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )

        self.acoustic_model = self.models[0]
        if torch.cuda.is_available():
            self.acoustic_model = self.acoustic_model.to(torch.device("cuda"))

        TTSHubInterface.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator(self.models, self.cfg)

    def __call__(self, x: str) -> (Optional[torch.Tensor], Optional[int]):
        with torch.no_grad():
            sample = TTSHubInterface.get_model_input(self.task, x)

            if torch.cuda.is_available():
                sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(torch.device("cuda"))
                sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(torch.device("cuda"))

            wav, rate = TTSHubInterface.get_prediction(self.task, self.acoustic_model, self.generator, sample)

            if torch.cuda.is_available():
                wav = wav.detach().cpu()

            gc.collect()
            return wav.unsqueeze(0), rate


class VietnameseSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self):
        pass

    def nat_normalize_text(self, text: str):
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()
        sil = FLAGS.special_phonemes[FLAGS.sil_index]
        text = re.sub(r"[\n.,:]+", f" {sil} ", text)
        text = text.replace('"', " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.,:;?!]+", f" {sil} ", text)
        text = re.sub("[ ]+", " ", text)
        text = re.sub(f"( {sil}+)+ ", f" {sil} ", text)
        
        return text.strip()

    def __call__(self, x: str):
        if len(x) > 500:
            x = x[:500]

        x = self.nat_normalize_text(x)
        mel = text2mel(
            x,
            "resources/lexicon.txt",
            0.2,
        )

        wave = mel2wave(mel)
        gc.collect()
        return (wave * (2 ** 15)).astype(np.int16), FLAGS.sample_rate


if __name__ == "__main__":
    worker = VietnameseSpeechSynthesizer()
    wav, rate = worker("Tôi là ai giữa cuộc đời này")
    print(wav.shape, rate)
    sounddevice.play(wav, samplerate=rate)
    sounddevice.wait()
    torchaudio.save("data/example_out_003.wav", wav, rate)
