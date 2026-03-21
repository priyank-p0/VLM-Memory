"""Gemini Vision adapter for image analysis with memory context"""
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_vlm(api_key=None, **kwargs) -> "GeminiVLM":
    return GeminiVLM(api_key=api_key, **kwargs)


class GeminiVLM:
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        from google import genai
        from visual_memory.config import get_config
        self.client = genai.Client(api_key=api_key or get_config().gemini_api_key or None)
        self.model = model

    def analyze(self, image_path: str, task: str = None) -> dict:
        return self.analyze_with_memory(image_path, [], task)

    def analyze_with_memory(self, image_path: str, memories: list, task: str = None) -> dict:
        try:
            from google.genai import types
            image_bytes, mime_type = self._prepare_image(image_path)
            memory_ctx = self._build_memory_prompt(memories, task)
            prompt = (
                "You are a vision analyst with persistent visual memory.\n\n"
                f"{memory_ctx}\n\n"
                "Respond ONLY in JSON (no markdown fences):\n"
                '{"description": "...", "tags": ["..."], "reasoning": "..."}'
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            return self._safe_parse(response.text)
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {"description": f"Error: {e}", "tags": [], "reasoning": ""}

    def _build_memory_prompt(self, memories: list[dict], task: str = None) -> str:
        if not memories:
            return "No prior visual memories."
        ctx = "Similar images you have seen before:\n"
        for i, m in enumerate(memories[:5]):
            desc = (m.get("description") or "")[:200]
            tags = ", ".join(m.get("tags") or [])
            ctx += f"\nMemory {i + 1} (sim: {m.get('similarity', 0):.2f}): {desc}\n"
            if tags:
                ctx += f"  Tags: {tags}\n"
        if task:
            ctx += f"\nTask: {task}\n"
        ctx += "\nReference specific memories in your reasoning."
        return ctx

    def _prepare_image(self, image_path: str) -> tuple[bytes, str]:
        suffix = Path(image_path).suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg"
        data = Path(image_path).read_bytes()
        if len(data) > 1_000_000:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            data = buf.getvalue()
            mime_type = "image/jpeg"
        return data, mime_type

    def _safe_parse(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"description": text[:500], "tags": [], "reasoning": ""}
