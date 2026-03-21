import duckdb
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS visual_memories (
    sample_id    VARCHAR NOT NULL,
    dataset_name VARCHAR DEFAULT 'default',
    filepath     VARCHAR,
    embedding    FLOAT[512] NOT NULL,
    description  VARCHAR,
    tags         VARCHAR[],
    session_id   VARCHAR,
    vlm_model    VARCHAR,
    metadata     VARCHAR,
    created_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (sample_id, dataset_name)
);
"""


class VisualMemoryDB:
    def __init__(self, db_path="memory.duckdb"):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self.con.execute("INSTALL vss; LOAD vss;")
        self.con.execute(SCHEMA)

    def write(self, sample_id, filepath, embedding, description=None,
              tags=None, session_id=None, dataset_name="default",
              vlm_model=None, metadata=None):
        try:
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            tags_list = tags or []

            self.con.execute("""
                INSERT OR REPLACE INTO visual_memories
                    (sample_id, dataset_name, filepath, embedding,
                     description, tags, session_id, vlm_model, metadata)
                VALUES (?, ?, ?, ?::FLOAT[512], ?, ?, ?, ?, ?)
            """, [
                sample_id, dataset_name, filepath, emb_list,
                description, tags_list, session_id, vlm_model,
                str(metadata) if metadata else None
            ])
            return True
        except Exception as e:
            logger.error(f"write failed: {e}")
            return False

    def search(self, embedding, k=5, dataset_name=None, session_id=None):
        try:
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            # Brute-force cosine similarity — fast enough for <10k rows
            query = """
                SELECT
                    sample_id, filepath, description, tags,
                    array_cosine_similarity(embedding, ?::FLOAT[512]) AS similarity
                FROM visual_memories
                WHERE 1=1
            """
            params = [emb_list]

            if dataset_name:
                query += " AND dataset_name = ?"
                params.append(dataset_name)
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY similarity DESC LIMIT ?"
            params.append(k)

            rows = self.con.execute(query, params).fetchall()
            columns = ["sample_id", "filepath", "description", "tags", "similarity"]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"search failed: {e}")
            return []

    def count(self, dataset_name=None):
        try:
            if dataset_name:
                r = self.con.execute(
                    "SELECT COUNT(*) FROM visual_memories WHERE dataset_name = ?",
                    [dataset_name]).fetchone()
            else:
                r = self.con.execute("SELECT COUNT(*) FROM visual_memories").fetchone()
            return r[0]
        except Exception:
            return 0

    def clear(self, dataset_name=None, session_id=None):
        try:
            if dataset_name and session_id:
                self.con.execute(
                    "DELETE FROM visual_memories WHERE dataset_name = ? AND session_id = ?",
                    [dataset_name, session_id])
            elif dataset_name:
                self.con.execute(
                    "DELETE FROM visual_memories WHERE dataset_name = ?", [dataset_name])
            else:
                self.con.execute("DELETE FROM visual_memories")
        except Exception as e:
            logger.error(f"clear failed: {e}")

    def export_parquet(self, output_path, dataset_name=None):
        """Export to Parquet for FiftyOne ingest. Person C uses this."""
        try:
            query = "SELECT * FROM visual_memories"
            params = []
            if dataset_name:
                query += " WHERE dataset_name = ?"
                params.append(dataset_name)
            self.con.execute(
                f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)",
                params if params else None)
            return True
        except Exception as e:
            logger.error(f"export_parquet failed: {e}")
            return False

    def close(self):
        self.con.close()
