import json
import joblib
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

from logic.config import Config
from logic.model.model_utils import preprocess_token, is_token_allowed


class ChatbotTrainer:
    def __init__(self):
        with open(Config.INTENTS_PATH, 'r') as f:
            self.intents = json.load(f)['intents']
        self.vectorizer = CountVectorizer(binary=True)
        self.encoder = LabelEncoder()
        self.nlp = spacy.load(Config.SPACY_LANGUAGE_PIPELINE)
        self.nlp.Defaults.stop_words -= Config.STOP_WORDS_TO_REMOVE

    def prepare_intents(self):
        """
        Parcours le fichiers 'intents', nettoie les données (vectorisation et lemmatisation)

        Return:
            - sentences (List) : les phrases nettoyées
            - tags (List) : les tags correspondants à chaque phrase
        """
        sentences, tags = [], []
        for intent in self.intents:
            tag = intent['tag']
            for pattern in intent["patterns"]:
                tokens = self.nlp(pattern)
                filtered_tokens = [
                    preprocess_token(token) 
                    for token in tokens 
                    if is_token_allowed(token)
                ]
                clean_sentence = " ".join(filtered_tokens)
                sentences.append(clean_sentence)
                tags.append(tag)
        return sentences, tags
        
    def vectorize_data(self, X, y):
        """
        Vectorisation (Bag of Words) et enregistrement.

        Args:
            - X (List[str]) : la liste à vectoriser
            - y (List[str]) : la liste des tags correspondants

        Returns:
            - X (array) : tableau de la vectorisation 
            - y (List) : les tags encodés
        """
        X = self.vectorizer.fit_transform(X).toarray()
        y = self.encoder.fit_transform(y)
        joblib.dump(self.vectorizer, f"./logic/model/{Config.VECTORIZER_NAME}.joblib")
        joblib.dump(self.encoder, f"./logic/model/{Config.ENCODER_NAME}.joblib")
        return X, y

    def train_model(self, X, y):
        """
        Définit, entraine sur X et y, et sauvegarde le modèle

        Args:
            - X (array) : données d'entrainement
            - y (List) : label d'entrainement
        """
        try:
            model = Sequential()

            model.add(Input(shape=(X.shape[1],)))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu')),
            model.add(Dense(len(y), activation='softmax'))

            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
            model.fit(X, y, epochs=100, batch_size=8, verbose=1)

            model.save(f"./logic/model/{Config.MODEL_NAME}.keras")
            return True
        
        except Exception as e:
            print("Erreur lors de l'entrainement du modèle", e)
            return False