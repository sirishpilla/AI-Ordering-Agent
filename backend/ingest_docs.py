import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
MODEL_NAME = "all-MiniLM-L6-v2"


# PostgreSQL connection helper
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "ai_ordering"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD")
    )



def chunk_text(text: str, chunk_size: int = 300):
    text = " ".join(text.split())
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def ensure_table():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source_file, chunk_index)
                );
                """
            )
        conn.commit()


def ingest_docs():
    if not DOCS_DIR.exists():
        print(f"Docs folder not found: {DOCS_DIR}")
        return

    files = list(DOCS_DIR.glob("*.txt"))

    if not files:
        print("No .txt files found.")
        return

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    ensure_table()

    with get_db_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cursor:
            total_chunks = 0

            for file_path in files:
                content = file_path.read_text(encoding="utf-8")
                chunks = chunk_text(content)

                print(f"\nProcessing {file_path.name}: {len(chunks)} chunk(s)")

                for index, chunk in enumerate(chunks, start=1):
                    embedding = model.encode(chunk).tolist()

                    cursor.execute(
                        """
                        INSERT INTO knowledge_chunks
                            (source_file, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_file, chunk_index)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding;
                        """,
                        (file_path.name, index, chunk, embedding),
                    )

                    total_chunks += 1

            conn.commit()

    print(f"\nIngestion complete. Stored/updated {total_chunks} chunks.")


if __name__ == "__main__":
    ingest_docs()