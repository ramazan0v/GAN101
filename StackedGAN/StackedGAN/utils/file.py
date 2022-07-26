import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

def save_modelXX(model, name):
    """
        model : Keras model
        name : String
    """
    json = model.to_json()
    with open(name, 'w') as f:
        f.write(json)


def load_modelXX(name):
    """
        name : String
        Keras model
    """
    with open(name) as f:
        json = f.read()
    model = model_from_json(json)
    return model

def load_complete_model(original_model, name):
    if os.path.isfile(name):
        try:
            original_model = load_model(name)
        except:
            original_model = original_model
    return original_model