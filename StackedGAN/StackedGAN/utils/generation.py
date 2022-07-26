import numpy as np


def generate(G, source):
    """
        G : Keras model
        source : List or Image
        images : Numpy array
    """
    input_dim = G.input_shape[1]
    images = G.predict(source)
    images = images * 255
    return images