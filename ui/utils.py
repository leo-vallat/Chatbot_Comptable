import json
import os
import streamlit as st

def initialize_session_state():
    """Initialise les valeurs du st.session_state au démarrage de l'app."""
    default_values = {
        "messages" : [],
        "kb_saved" : False,
        "model_trained" : False,
        "model_reloaded" : False
    }
    
    for k, v in default_values.items():
        if k not in st.session_state:
            st.session_state[k] = v

def update_navigation():
    """Met la valeur de st.session_state.navigation à la valeur de menu_value"""
    st.session_state.navigation = st.session_state.menu_value

def load_json_file(json_path):
    """Charge le contenu du json."""
    try: 
        os.path.exists(json_path)
    except Exception as e:    
        print(f"Error while loading the json : {e}")
    else:
        with open(json_path, "r") as file:
            return json.load(file)

def save_in_json(json_path, data):
    """Dump data dans le json."""
    try:
        os.path.exists(json_path)
    except Exception as e:
        print(f"Error while trying to save in a json : {e}")
    else:
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)