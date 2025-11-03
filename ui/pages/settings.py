import json
import requests
import streamlit as st

from logic.config import Config
from logic.model.train_model import ChatbotTrainer
from ui.utils import initialize_session_state, load_json_file, save_in_json

def render_top_of_page():
    """Affiche les widgets du haut de la page."""
    st.header("🧠 Base de connaissance du chatbot", )

    st.space("small")
    # Style pour agrandir les dialogs
    st.markdown("""
        <style>
        div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
            width: 80vw;
        }
        </style>
    """, unsafe_allow_html=True)

    _, col_retrain_btn, _, col_add_btn, _ = st.columns([3, 3, 1, 3, 3])
    col_retrain_btn.button("Entrainer le modèle", icon="🚀", use_container_width=True, on_click=retrain_model)
    col_add_btn.button("Ajouter un tag", icon="➕", use_container_width=True, on_click=open_tag_dialog)

    st.space('small')
    st.markdown("*Visualisez et modifiez les intents (tags, patterns, responses).*")



def render_intents():
    """Affiche les intents et les boutons pour les modifier / supprimer."""
    intents = load_json_file(Config.INTENTS_PATH)['intents']
    
    if not intents:
        st.warning('Aucune données trouvées !')
        return
    
    for i, intent in enumerate(intents):
        st.markdown(f"### 🏷️ {intent['tag']}")

        with st.expander('🧩 Patterns', expanded=False):
            for pattern in intent['patterns']:
                st.markdown(f"- {pattern}")
        
        with st.expander('💬 Responses', expanded=False):
            for response in intent['responses']:
                st.markdown(f"- {response}")

        with st.container(horizontal=True):
            st.button("✏️", key=f"edit_intent_btn_{i}", type='tertiary', on_click=open_tag_dialog, args=('edit', intent))
            st.button("🗑️", key=f"delete_intent_btn_{i}", type='tertiary', on_click=delete_intent, args=[intent['tag']])
            st.space('stretch')

@st.dialog("Modifier la base de connaissance")
def open_tag_dialog(mode='create', intent=None):
    """Ouvre une pop up pour créer ou modifier un intent."""
    st.html("<span class='big-dialog'></span>")

    if mode=='edit' and intent:
        st.session_state['tag_name'] = intent['tag']
        st.session_state['patterns'] = "\n".join(intent.get("patterns", []))
        st.session_state['responses'] = "\n".join(intent.get("responses", []))
    else:
        st.session_state['tag_name'] = ""
        st.session_state['patterns'] = ""
        st.session_state['responses'] = ""

    tag_name = st.text_input("Nom du tag", value=st.session_state['tag_name'])
    patterns = st.text_area("Patterns (un par ligne)", value=st.session_state['patterns'])
    responses = st.text_area("Responses (une par ligne)", value=st.session_state['responses'])

    save_btn_col, cancel_btn_col = st.columns([1, 1])
    with save_btn_col:
        save = st.button("💾 Sauvegarder", on_click=save_intent, args=(tag_name, patterns, responses, mode, intent))
    with cancel_btn_col:
        cancel = st.button("❌ Annuler")
    
    if save:
        st.session_state.kb_saved = True
    if save or cancel:
        st.rerun()

def save_intent(tag_name, patterns, responses, mode, old_intent=None):
    """Sauvegarder les intents dans le fichier intents.json"""
    intents = load_json_file(Config.INTENTS_PATH)['intents']

    new_intent = {
        "tag": tag_name.strip(),
        "patterns": [p.strip() for p in patterns.split("\n") if p.strip()],
        "responses": [r.strip() for r in responses.split("\n") if r.strip()]
    }

    if mode == "edit" and old_intent:
        for idx, intent in enumerate(intents):
            if intent['tag'] == old_intent['tag']:
                intents[idx] = new_intent
                break
    else:
        intents.append(new_intent)

    data = {'intents': intents}
    save_in_json(Config.INTENTS_PATH, data)

    if Config.AUTO_RETRAIN_MODEL:
        retrain_model()    

def delete_intent(tag):
    """Supprime un intent du fichier."""
    intents = load_json_file(Config.INTENTS_PATH)['intents']
    intents = [i for i in intents if i['tag'] != tag]
    data = {'intents': intents}
    save_in_json(Config.INTENTS_PATH, data)

    if Config.AUTO_RETRAIN_MODEL:
        retrain_model()
    
    st.toast(f"✅ Tag '{tag}' supprimé !")

def retrain_model():
    """Lance un entrainement du réseau de neurone et le recharge."""
    try:
        trainer = ChatbotTrainer()
        sentences, tags = trainer.prepare_intents()
        X_train, y_train = trainer.vectorize_data(sentences, tags)
        success = trainer.train_model(X_train, y_train)
        if success:
            st.session_state.model_trained = True
            reload_model()

    except Exception as e:
        st.warning("⚠️ Erreur lors de l'entrainement du modèle !")
        print(e)
    
def reload_model():
    """Appel l'endpoint 'reload model' pour recharger le nouveau modèle."""
    try:
        success = requests.post(f'{Config.API_URL}/reload_model')
        if success.status_code == 200:
            st.session_state.model_reloaded = True
    except Exception as e:
        st.warning(f"⚠️ Impossible de recharger le modèle")
        print(e)

def display_messages():
    """Affiche les messages de succès."""
    if st.session_state.kb_saved:
        st.toast("✅ Base de connaissance mise à jour !")
        st.session_state.kb_saved = False
    if st.session_state.model_trained:
        st.toast("✅ Entrainement du modèle effectué !")
        st.session_state.model_trained = False
    if st.session_state.model_reloaded:
        st.toast("✅ Modèle rechargé côté serveur !")
        st.session_state.model_reloaded = False

def main():
    """"""
    initialize_session_state()

    render_top_of_page()
    render_intents()

    display_messages()