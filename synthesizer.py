import gc
import torch
import torchaudio

from typing import Optional
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


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


# if __name__ == "__main__":
#     worker = EnglishSpeechSynthesizer()
#     wav, rate = worker("That morning I saw you crying")
#     torchaudio.save("data/example_out_003.wav", wav, rate)
