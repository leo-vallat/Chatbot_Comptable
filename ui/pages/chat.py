import requests
import streamlit as st
from logic.config import Config
from ui.utils import initialize_session_state

def display_conversation():
    """"""
    with st.container(border=True, height=600):
        if not st.session_state.messages:
            st.write("💬 Démarrez une nouvelle conversation !")
            return
        for role, text in st.session_state.messages:
            with st.chat_message(role):
                st.write(text)

def display_input():
    """"""
    col1, col2 = st.columns([12,1])
    with col1:
        st.chat_input('Écrivez un message ...', key='chat_input', on_submit=process_submitting)
    with col2:
        st.button("🗑️", key='reset_btn', on_click=reset_conversation)

def process_submitting():
    """"""
    prompt = st.session_state.chat_input.strip()
    if not prompt:
        return
    
    st.session_state.messages.append(('user', prompt))
    
    answer = call_api(prompt)
    st.session_state.messages.append(('assistant', answer))

def call_api(prompt):
    """"""
    try:
        response = requests.post(f'{Config.API_URL}/process_prediction', json={'text': prompt})
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response')
        else:
            answer = 'Erreur de communication avec le serveur 😕'
    except Exception as e:
        answer = f'Erreur : {e}'

    return answer


def reset_conversation():
    """"""
    st.session_state.messages = []

def main():
    """"""
    initialize_session_state()

    st.header("Budget - le chatbot expert en comptabilité")

    display_conversation()

    display_input()