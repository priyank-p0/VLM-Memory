import sys
import os
sys.path.insert(0, ".")

from visual_memory.seed import seed

if __name__ == "__main__":
    print("Starting seeding process with n=5...")
    # Use mocks if no API key
    seed(db_path="memory.duckdb", dataset_name="demo", n=5)
    print("Seeding test complete.")
