class Config:
    API_URL = "http://127.0.0.1:5120"
    
    SPACY_LANGUAGE_PIPELINE = 'fr_core_news_sm'
    STOP_WORDS_TO_REMOVE = {"tu", "es", "ce", "que", "quoi", "quel", "quelle", "qu'", "est"}

    INTENTS_PATH = "logic/data/intents.json"

    MODEL_NAME = 'chatbrain'
    VECTORIZER_NAME = 'vectorizer'
    ENCODER_NAME = 'encoder'

    AUTO_RETRAIN_MODEL = True