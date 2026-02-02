
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
# This is required to make psycopg2 understand and correctly handle the VECTOR type
import pgvector.psycopg2

# --- Configuration: ⚠️ CONFIRM THESE DETAILS ⚠️ ---
DB_HOST = "localhost"
DB_NAME = "vigilance_bot"  # Must match your database name
DB_USER = "postgres"  # Your PostgreSQL username
DB_PASS = "lekshmihr@12"  # **REPLACE THIS WITH YOUR ACTUAL PASSWORD**

# Initialize the embedding model once. 'all-MiniLM-L6-v2' is small and fast.
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings_and_store():
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        pgvector.psycopg2.register_vector(conn)
        cur = conn.cursor()

        # 1. Fetch text from the NEW content_nodes table
        # We join with 'sections' so the AI knows WHICH section the text belongs to
        sql_fetch = """
            SELECT cn.id, s.sec_no, s.title, cn.text_content
            FROM content_nodes cn
            JOIN sections s ON cn.section_id = s.id
            WHERE cn.is_active = TRUE;
        """
        cur.execute(sql_fetch)
        records = cur.fetchall()

        # 2. Build a rich context string for the AI to understand
        node_ids = [r[0] for r in records]
        texts = [f"{r[1]} - {r[2]}. Content: {r[3]}" for r in records]

        # 3. Generate 384-dimension embeddings
        print(f"Generating embeddings for {len(texts)} vigilance nodes...")
        embeddings = model.encode(texts, convert_to_tensor=False)

        # 4. Update the content_nodes table directly
        # Instead of a new table, we fill the empty 'embedding' column
        update_sql = "UPDATE content_nodes SET embedding = %s WHERE id = %s"
        
        for node_id, emb in zip(node_ids, embeddings):
            cur.execute(update_sql, (emb.tolist(), node_id))

        conn.commit()
        print(f"✅ Successfully updated {len(node_ids)} nodes in 'content_nodes'.")

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()


if __name__ == "__main__":
    generate_embeddings_and_store()