import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from typing import List
from src.core.config import settings

class MultimodalCaptioner:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(settings.CAPTION_MODEL).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(settings.CAPTION_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.CAPTION_MODEL)
        self.model.eval()

    def caption_images(self, images: List[Image.Image], max_new_tokens: int = 50) -> List[str]:
        if not images:
            return []
        try:
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
            captions = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]
            return captions
        except Exception as e:
            return [f"Diagram/image extraction note (captioning failed: {str(e)})" for _ in images]
