"""End-to-end smoke test for the visual memory pipeline."""
import glob
import sys

import numpy as np


def find_test_image() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "**/*.jpg", "**/*.jpeg", "**/*.png"]
    images = []
    for pattern in patterns:
        images += glob.glob(pattern, recursive=True)
    if not images:
        print("No JPEG or PNG found. Pass an image path as argument.")
        sys.exit(1)
    return images[0]


def main():
    image_path = find_test_image()
    print(f"Using image: {image_path}\n")

    # --- Encoder ---
    from visual_memory.encoder import get_encoder
    encoder = get_encoder()

    img_emb = encoder.encode_image(image_path)
    assert img_emb is not None, "encode_image returned None"
    assert img_emb.shape == (512,), f"Expected (512,), got {img_emb.shape}"

    txt_emb = encoder.encode_text("a photo of a dog")
    assert txt_emb.shape == (512,), f"Expected (512,), got {txt_emb.shape}"

    similarity = float(np.dot(img_emb, txt_emb))
    print(f"Cosine similarity (image vs 'a photo of a dog'): {similarity:.4f}")

    # --- VLM: no memories ---
    from visual_memory.vlm_adapter import get_vlm
    vlm = get_vlm()

    print("\n[1] Analyzing image with no memories...")
    result_no_mem = vlm.analyze(image_path)
    print(f"  description : {result_no_mem.get('description', '')[:120]}")
    print(f"  tags        : {result_no_mem.get('tags')}")
    print(f"  reasoning   : {result_no_mem.get('reasoning', '')[:120]}")
    for key in ("description", "tags", "reasoning"):
        assert key in result_no_mem, f"Missing key '{key}' in result_no_mem"

    # --- VLM: with fake memories ---
    fake_memories = [
        {"description": "a dog running in grass", "tags": ["dog", "outdoor"], "similarity": 0.85},
        {"description": "sunset over a lake", "tags": ["landscape", "water"], "similarity": 0.71},
    ]

    print("\n[2] Analyzing image with 2 fake memories...")
    result_with_mem = vlm.analyze_with_memory(
        image_path,
        fake_memories,
        task="Describe the image and reference any relevant past observations",
    )
    print(f"  description : {result_with_mem.get('description', '')[:120]}")
    print(f"  tags        : {result_with_mem.get('tags')}")
    print(f"  reasoning   : {result_with_mem.get('reasoning', '')[:120]}")
    for key in ("description", "tags", "reasoning"):
        assert key in result_with_mem, f"Missing key '{key}' in result_with_mem"

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
