import torch

from ...training.train import (
    load_model,
    predict as cnn_predict,
    resolve_hint_color,
)
from ...utils.logger import setup_logger
from src.utils.config.paths import ROOT_DIR

class ColorCNNBooster:
    def __init__(self, config, log_level):
        self.logger = setup_logger(__name__, level=log_level)
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = ROOT_DIR / config.vision_config.color_classifier_weights
        self.cnn, self.cnn_idx2color = load_model(model_path, self.device)
        self._cnn_abstain_threshold = config.features.cnn_abstain_below_confidence

        self.logger.info(f"Color CNN loaded from {model_path}")

    def predict_color(self, det, frame):
        """Try to assign color via CNN. Returns BGR tuple or None on failure."""
        try:
            crop = det.box.as_int().cut(frame)
            if crop.size == 0:
                self.logger.debug(f"CNN assertion failed: crop size is 0")
                return None
            idx, conf = cnn_predict(self.cnn, crop, self.device)
            ui_colors = list(self.config.features.colors)
            color_name = resolve_hint_color(
                idx,
                conf,
                self.cnn_idx2color,
                ui_colors,
                unknown_threshold=self._cnn_abstain_threshold,
            )
            self.logger.debug(f"CNN: id={det.id} → {color_name} ({conf:.2f})")
            return color_name
        except Exception as e:
            self.logger.debug(f"CNN error: {e}")
            return None
