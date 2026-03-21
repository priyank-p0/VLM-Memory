"""CLIP encoder for image and text embeddings"""
import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)
_instance = None


def get_encoder(model_name="ViT-B-32", pretrained="openai") -> "CLIPEncoder":
    global _instance
    if _instance is None:
        _instance = CLIPEncoder(model_name, pretrained)
    return _instance


class CLIPEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        try:
            import open_clip
            import torch
            self._torch = torch
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            self._mock = False
        except Exception as e:
            logger.warning(f"CLIP unavailable ({e}), using deterministic mock embeddings")
            self._mock = True

    def encode_image(self, image_path: str) -> np.ndarray | None:
        if not os.path.exists(image_path):
            return None
        if self._mock:
            return self._deterministic_mock(image_path)
        try:
            from PIL import Image
            img = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
            with self._torch.no_grad():
                feat = self.model.encode_image(img)
                feat /= feat.norm(dim=-1, keepdim=True)
            return feat.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning(f"encode_image failed for {image_path}: {e}")
            return None

    def encode_text(self, text: str) -> np.ndarray:
        if self._mock:
            return self._deterministic_mock(text)
        tokens = self.tokenizer([text])
        with self._torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy().astype(np.float32)

    def _deterministic_mock(self, seed_input) -> np.ndarray:
        h = int(hashlib.sha256(str(seed_input).encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % 2**31)
        vec = rng.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
