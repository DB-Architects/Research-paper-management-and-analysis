import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

# --- CONFIG ---
DB_PARAMS = {
    "dbname": "dblp_project",
    "user": "siddharthkonnur",
    "password": "",   # your WSL postgres password
    "host": "localhost",
    "port": "5432"
}

BATCH_SIZE = 512 # papers per batch
LIMIT = 100000 

def generate_embeddings():
    print("Loading model... (first time downloads ~90MB)")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='mps')
    print("Model loaded!")

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Fetch papers that don't have embeddings yet
    print(f"Fetching up to {LIMIT} papers without embeddings...")
    cur.execute("""
        SELECT p.paper_id, p.abstract_text
        FROM Papers p
        LEFT JOIN Paper_Embeddings pe ON p.paper_id = pe.paper_id
        WHERE pe.paper_id IS NULL
          AND p.abstract_text IS NOT NULL
          AND p.abstract_text != ''
        LIMIT %s
    """, (LIMIT,))

    rows = cur.fetchall()
    print(f"Found {len(rows)} papers to embed.")

    if not rows:
        print("All papers already have embeddings!")
        return

    # Process in batches
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        paper_ids = [r[0] for r in batch]
        abstracts = [r[1] for r in batch]

        # Generate embeddings
        embeddings = model.encode(abstracts, show_progress_bar=False)

        # Insert into DB
        data = [
            (paper_ids[j], embeddings[j].tolist())
            for j in range(len(batch))
        ]

        execute_values(cur, """
            INSERT INTO Paper_Embeddings (paper_id, embedding)
            VALUES %s
            ON CONFLICT (paper_id) DO NOTHING
        """, data, template="(%s, %s::vector)")

        conn.commit()
        total += len(batch)
        print(f"Embedded {total}/{len(rows)} papers...")

    print(f"\nDone! {total} papers embedded successfully.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    generate_embeddings()