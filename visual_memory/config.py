import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    gemini_api_key: str = ""
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    db_path: str = "memory.duckdb"


def get_config() -> Config:
    return Config(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
    )
