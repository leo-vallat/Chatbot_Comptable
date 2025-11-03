import streamlit as st
from ui.utils import initialize_session_state, update_navigation

st.set_page_config(
    page_title="Bud-get le chatbot comptabilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_session_state()

st.sidebar.title("Menu")

st.sidebar.radio(
    " ", 
    ["Chat", "Réglages"], 
    index=0, 
    on_change=update_navigation,
    key="menu_value"
)

if st.session_state.menu_value == "Chat":
    from ui.pages.chat import main as render_chat
    render_chat()
elif st.session_state.menu_value == "Réglages":
    from ui.pages.settings import main as render_settings
    render_settings()

