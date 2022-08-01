from typing import Optional
from transformers import AutoTokenizer, AutoModel
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM


class Translator(object):
    def __call__(self, x: str) -> Optional[str]:
        pass


class LanguageTranslatorBART(Translator):
    def __init__(self, model_path="weights/trans/vi2en", target_prefix="en"):
        super(LanguageTranslatorBART, self).__init__()
        self.target_prefix = f"{target_prefix} "
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.network = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def __call__(self, x: str) -> Optional[str]:
        x = f"</s>{x}</s>"
        x = self.tokenizer(x, return_tensors="pt")
        x = self.network.generate(**x)
        return self.tokenizer.decode(x[0], skip_special_tokens=True).replace(self.target_prefix, "")


class LanguageTranslator(Translator):
    def __init__(self, cmd="vi"):
        super(LanguageTranslator, self).__init__()
        if cmd == "vi":
            model_alias = "vinai/vinai-translate-vi2en"
            source_lang = "vi_VN"
        else:
            model_alias = "vinai/vinai-translate-en2vi"
            source_lang = "en_XX"

        self.source_lang = source_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias, source_lang=source_lang)
        self.network = AutoModelForSeq2SeqLM.from_pretrained(model_alias)

    def __call__(self, x: str) -> Optional[str]:
        input_ids = self.tokenizer(x, return_tensors="pt").input_ids
        output_ids = self.network.generate(
            input_ids,
            do_sample=True,
            top_k=100,
            top_p=0.8,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.source_lang],
            num_return_sequences=1
        )

        target_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        target_text = " ".join(target_text)

        return target_text


if __name__ == "__main__":
    translator = LanguageTranslator(cmd="en")
    print(translator("It is also possible to leverage this 1-D computation to also compute efficiently OT on the circle "
                     "as shown by Delon et al. [2010]. Note that if the cost is a concave function of the distance, "
                     "notably when p < 1, the behavior of the optimal transport plan is very different, yet efficient "
                     "solvers also exist [Delon et al., 2012]."))

