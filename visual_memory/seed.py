"""Pre-populate DuckDB memory database using the encoder and VLM."""
import sys

from visual_memory.memory_db import VisualMemoryDB

from visual_memory.encoder import get_encoder
from visual_memory.vlm_adapter import get_vlm


def seed(db_path="memory.duckdb", dataset_name="demo", n=30):
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset("quickstart")
    samples = list(dataset.take(n))

    encoder = get_encoder()
    vlm = get_vlm()
    db = VisualMemoryDB(db_path)

    written = 0
    for i, sample in enumerate(samples, start=1):
        filepath = sample.filepath
        print(f"[{i}/{n}] {filepath}")

        embedding = encoder.encode_image(filepath)
        if embedding is None:
            print(f"  skipping — encoding failed")
            continue

        result = vlm.analyze(filepath, task="Describe this image in detail")

        db.write(
            sample_id=str(sample.id),
            filepath=filepath,
            embedding=embedding,
            description=result.get("description", ""),
            tags=result.get("tags", []),
            dataset_name=dataset_name,
            vlm_model="gemini-2.5-flash",
            session_id="seed",
        )
        written += 1

    print(f"\nDone. {written}/{n} images written to {db_path}")


if __name__ == "__main__":
    seed()
