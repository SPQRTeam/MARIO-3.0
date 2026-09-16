import cv2
from PIL import Image
import torch
from torchvision import transforms
import yaml

from src.ml.models.color_classifier import ColorClassifierCNNv0
from ...utils.logger import setup_logger
from src.utils.config.paths import ROOT_DIR

class ColorCNNNao:
    def __init__(self, config, log_level):
        """Load CNN color classification model."""
        self.logger = setup_logger(__name__, level=log_level)
        config_path = ROOT_DIR / config.vision_config.color_classifier_config

        if not config_path.exists():
            raise FileNotFoundError(f"Color classifier config not found: {config_path}")

        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        self.idx2color = {int(k): v for k, v in model_config['idx2label'].items()}
        num_classes = model_config['num_classes']
        image_size = tuple(model_config['image_size'])
        model_path = ROOT_DIR / config.vision_config.color_classifier_weights

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_model = ColorClassifierCNNv0(in_channels=3, num_classes=num_classes)
        self.color_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.color_model.to(self.device)
        self.color_model.eval()

        # Transform
        self.color_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.logger.info(f"Color model loaded on {self.device}")

    def predict_color(self, det, frame) -> str:
        """
        Predict robot color from image crop.

        Args:
            robot_img: Robot bounding box crop (BGR).

        Returns:
            Predicted color name.
        """

        try:
            crop = det.box.as_int().cut(frame)
            if crop.size == 0:
                self.logger.debug(f"CNN assertion failed: crop size is 0")
                return None
            # Convert BGR to RGB
            robot_img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(robot_img_rgb)
            image_tensor = self.color_transform(image_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.color_model(image_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_color = self.idx2color[pred_idx]

            return pred_color
        except Exception as e:
            self.logger.warning(f"Color prediction error: {e}")
            return None
