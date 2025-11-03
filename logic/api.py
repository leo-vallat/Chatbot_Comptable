import joblib
import json
import jwt
import numpy as np
import os
import random
import spacy
import traceback
from datetime import datetime, timedelta, UTC
from flask import Flask, jsonify, request
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

from logic.config import Config
from logic.model.model_utils import preprocess_token, is_token_allowed

class ChatbotAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self._register_routes()

        self.nlp = spacy.load(Config.SPACY_LANGUAGE_PIPELINE)
        self.nlp.Defaults.stop_words -= Config.STOP_WORDS_TO_REMOVE

        self.load_ressources()

    def load_ressources(self):
        """"""
        try:
            self.vectorizer = joblib.load(f"./logic/model/{Config.VECTORIZER_NAME}.joblib")
            self.encoder = joblib.load(f"./logic/model/{Config.ENCODER_NAME}.joblib")
            self.model = load_model(f"./logic/model/{Config.MODEL_NAME}.keras")
            with open(Config.INTENTS_PATH, 'r') as f:
                self.intents = json.load(f)['intents'] 

            print(f"Nombre d'intents chargés : {len(self.intents)}")
            print(f"Classes reconnues par le modèle : {len(self.encoder.classes_)}")

            return True
        
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            traceback.print_exc()
            return False


    def _register_routes(self):
        """Enregistre les endpoints de l'API"""
        self.app.add_url_rule('/health', 'health_check', self.health_check, methods=['GET'])
        self.app.add_url_rule('/process_prediction', 'process_prediction', self.process_prediction, methods=['POST'])
        self.app.add_url_rule('/reload_model', 'reload_model', self.reload_model, methods=['POST'])

    def health_check(self):
        """Endpoint pour le service de test de santé"""
        return jsonify({
            "status": "OK",
            "message": "Service operational",
            "timestamp": datetime.now(UTC).isoformat()
        }), 200

    def process_prediction(self):
        """Endpoint pour exécuter le chatbot"""
        try:
            payload = request.get_json()
            text = payload.get("text", None)
            if not text:
                return jsonify({"msg": "Missing 'text' parameter"}), 400
            text = self._prepare_data(text)
            array = self._vectorize_data(text)      
            intent, prob = self._predict_intent(array)
            response = self._get_response(intent, prob)
            return jsonify(response), 200

        except KeyError as e:
            return jsonify({'msg': f'Missing required parameter: {e}'}), 400
        except Exception as e:
            return jsonify({'msg': 'Internal server error'}), 500

    def _prepare_data(self, text):
        """Nettoie les données (vectorisation, lemmatisation, etc)."""
        tokens = self.nlp(text)
        filtered_tokens = [
            preprocess_token(token) 
            for token in tokens 
            if is_token_allowed(token)
        ]
        return " ".join(filtered_tokens)

    def _vectorize_data(self, text):
        """Vectorisation (Bag of Words)"""
        return self.vectorizer.transform([text]).toarray()

    def _predict_intent(self, X):
        """Prédit l'intention du texte donné."""
        y = self.model.predict(X)
        predicted_class = np.argmax(y, axis=1)[0]
        predicted_tag = self.encoder.inverse_transform([predicted_class])

        print("prediction : ", predicted_tag)
        print("probabilité : ", float(np.max(y)))
        return predicted_tag, float(np.max(y))

    def _get_response(self, predicted_intent, prob):
        """Renvoi la réponse selon l'intent prédit."""
        for intent in self.intents:
            if intent['tag'] == predicted_intent:
                if prob > 0.6:
                    response = random.choice(intent['responses'])
                    return {
                        'tag': intent['tag'],
                        'response': response,
                        'confidence': prob
                    }
        return {
            'tag': None,
            'response': "Désole, je n'ai pas compris votre requête.",
            'confidence': prob
        }

    def reload_model(self):
        success = self.load_ressources()

        if success:
            return jsonify({"status": "ok", "message": "Modèle et ressources rechargés avec succès !"}), 200
        else:
            return jsonify({"status": "error", "message": "Échec lors du chargement du modèle"}), 500
    
    def run(self):
        self.app.run(host="0.0.0.0", port=5120, debug=True)

if __name__=='__main__':
    api = ChatbotAPI()
    api.run()