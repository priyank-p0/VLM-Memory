from visual_memory.memory_db import VisualMemoryDB
import numpy as np
import os

def test_memory_db():
    db_path = "test_memory.duckdb"
    parquet_path = "test_export.parquet"
    
    # Ensure clean state
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(parquet_path):
        os.remove(parquet_path)
        
    db = VisualMemoryDB(db_path)
    
    print("Testing write...")
    # Write sample 1
    db.write("s1", "/path/to/img1.jpg", np.random.rand(512).astype(np.float32),
             description="a dog", tags=["dog", "outdoor"], dataset_name="test_ds")
    
    # Write sample 2
    db.write("s2", "/path/to/img2.jpg", np.random.rand(512).astype(np.float32),
             description="a cat", tags=["cat", "indoor"], dataset_name="test_ds")
    
    # Upsert sample 1
    db.write("s1", "/path/to/img1.jpg", np.random.rand(512).astype(np.float32),
             description="updated dog", tags=["dog", "outdoor", "golden"], dataset_name="test_ds")
    
    print(f"Total entries in 'test_ds': {db.count('test_ds')}")
    assert db.count("test_ds") == 2
    
    print("Testing search...")
    query_emb = np.random.rand(512).astype(np.float32)
    results = db.search(query_emb, k=2, dataset_name="test_ds")
    
    assert len(results) == 2
    assert "similarity" in results[0]
    assert isinstance(results[0], dict)
    print(f"Search results: {results}")
    
    print("Testing export...")
    success = db.export_parquet(parquet_path, dataset_name="test_ds")
    assert success
    assert os.path.exists(parquet_path)
    print(f"Exported to {parquet_path}")
    
    db.close()
    
    # Cleanup
    os.remove(db_path)
    os.remove(parquet_path)
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_memory_db()
