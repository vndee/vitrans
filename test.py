# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# article_en = "The head of the United Nations says there is no military solution in Syria"
# article_en = "The erosion of mosses on the surface of Hue imperial citadel has caused a lot of harm and is currently an urgent problem to be solved."
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
#
# model_inputs = tokenizer(article_en, return_tensors="pt")
#
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"]
# )
# target = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#
# print(target)
#

from datasets import load_dataset
dataset = load_dataset("json", data_files="data/dev/en2vi/train.json")
print(dataset)
