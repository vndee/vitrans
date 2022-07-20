from googletrans import Translator

translator = Translator(
    service_urls = [
        "translate.google.com",
        "translate.google.com.vn"
    ]
)

if __name__ == "__main__":
    translator = Translator()
    print(translator.translate("Anh muốn được cùng em, về vùng biển vắng."))
