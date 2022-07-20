from typing import Optional
from transformers import AutoTokenizer, AutoModel
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM


class Translator(object):
    def __call__(self, x: str) -> Optional[str]:
        pass


class LanguageTranslator(Translator):
    def __init__(self, model_path="weights/trans/vi2en", target_prefix="en"):
        super(LanguageTranslator, self).__init__()
        self.target_prefix = f"{target_prefix} "
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.network = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def __call__(self, x: str) -> Optional[str]:
        x = f"</s>{x}</s>"
        x = self.tokenizer(x, return_tensors="pt")
        x = self.network.generate(**x)
        return self.tokenizer.decode(x[0], skip_special_tokens=True).replace(self.target_prefix, "")


# if __name__ == "__main__":
#     translator = LanguageTranslator()
#     print(translator("buổi sáng hôm ấy thấy em chợt khóc"))

