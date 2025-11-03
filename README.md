# Chatbot Expert Comptable

Un chatbot intelligent spécialisé dans la **comptabilité et la finance**, capable de :
- Répondre à des questions comptables (bilan, amortissement, TVA, etc.)
- Être enrichi via une interface Streamlit pour modifier sa base de connaissance
- Se réentraîner automatiquement après chaque modification du fichier `intents.json`

## Structure du projet
Le projet se compose d'un dossier logic - contenant l'ensemble du backend - et d'un fichier ui - contenant le code de l'interface graphique. La base de connaissance est un fichier json. Un fichier de configuration permet de modifier les variables importantes du logiciel.

| Composant            | Outil              |
| -----------------    | ------------------ |
| Backend              | Flask              |
| Frontend             | Streamlit          |
| NLP                  | spaCy              |
| Vectorisation        | Scikit-learn       |
| Sérialisation        | Joblib             |
| Modélisation         | Keras / TensorFlow |
| Base de connaissance | JSON               |

## Fonctionnalités
### Chat intéractif
- Interface graphique Streamlit
- Envoi d'un message à l'API Flask
- Reception d'une réponse du modèle

### Base de connaissance éditable
- Ajout, modification ou suppression des intents (tags, patterns, reponses) grâce à l'interface graphique
- Sauvegarde automatique dans le fichier 'intents.json'

### Réentrainement automatique
- Paramètre dans le fichier config pour activer/désactiver le réentrainement automatique
- Activé par défaut lors de la modification de intents.json
- le modèle (.keras), l'encodeur et le vectoriseur sont mis à jour.

## Installation
### Cloner le dépot Git 
````
git clone 
````
### Créer un environnement virtuel
````
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
````
### Installer les dépendances 
````
pip install -r requirements.txt
````

## Utilisation
### Lancement du backend
````
python -m logic.api
````
### Lancement de l'interface graphique
````
streamlit run app.py
````

