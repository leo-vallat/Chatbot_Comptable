def preprocess_token(token):
    """
    Lemmatise, supprime les espaces inutiles et met en minuscule le token

    Args:
        - token (str)
    
    Returns:
        - token traité (str)
    """
    return token.lemma_.strip().lower()

def is_token_allowed(token):
    """
    test la nature du token

    Arg: 
        - token (str)
    
    Return:
        - bool : True si le token est composé de caractères alphanumérique, False sinon
    """
    return bool(
        token
        and not token.is_space
        and not token.is_punct
        and not token.is_stop
    )