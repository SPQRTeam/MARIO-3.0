from .color_cnn_nao import ColorCNNNao
from .color_cnn_booster import ColorCNNBooster

class NullColorCNN:
    def __init__(self, *_, **__):
        pass

    def predict_color(self, *_, **__):
        return None
