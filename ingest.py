import os
import json
import sqlite3
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = "data"
PDFS_FOLDER = os.path.join(DATA_PATH, "industrial-safety-pdfs")
SOURCES_FILE = os.path.join(DATA_PATH, "sources.json")
DB_FILE = "chunks.db"
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "safety_docs"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- Text Chunking Parameters ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def setup_database():
    """Initializes the SQLite database and creates the chunks table with FTS5."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Drop table if it exists to ensure a fresh start
    cursor.execute("DROP TABLE IF EXISTS chunks")
    # Create a simple table for chunks
    cursor.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source_id TEXT NOT NULL,
            chunk_num INTEGER NOT NULL
        )
    """)
    # Create a virtual FTS5 table for full-text search
    cursor.execute("DROP TABLE IF EXISTS chunks_fts")
    cursor.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='rowid'
        );
    """)
    # Create a trigger to keep the FTS table in sync with the chunks table
    cursor.execute("""
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
    """)
    conn.commit()
    conn.close()
    print("✅ SQLite database and FTS5 table created successfully.")

def process_and_embed():
    """
    Processes PDFs, chunks text, creates embeddings, and stores them in 
    SQLite and ChromaDB.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    print("✅ Embedding model loaded.")

    print("Setting up ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        if CHROMA_COLLECTION in [c.name for c in chroma_client.list_collections()]:
            chroma_client.delete_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        print(f"Warning during Chroma collection cleanup: {e}")
        
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    print("✅ ChromaDB collection ready.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    with open(SOURCES_FILE, 'r') as f:
        sources = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("\nProcessing and embedding documents...")
    all_chunks_for_chroma = {
        'ids': [],
        'documents': [],
        'metadatas': []
    }

    for source in tqdm(sources, desc="Documents"):
        pdf_path = os.path.join(DATA_PATH, source['path'].split('/', 1)[1])
        if not os.path.exists(pdf_path):
            print(f"⚠️ Warning: File not found at {pdf_path}. Skipping.")
            continue
        
        try:
            reader = PdfReader(pdf_path)
            full_text = "".join(page.extract_text() or "" for page in reader.pages)
            chunks = text_splitter.split_text(full_text)

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{source['id']}-{i}"
                
                cursor.execute(
                    "INSERT INTO chunks (id, content, source_id, chunk_num) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk_text, source['id'], i)
                )

                all_chunks_for_chroma['ids'].append(chunk_id)
                all_chunks_for_chroma['documents'].append(chunk_text)
                all_chunks_for_chroma['metadatas'].append({
                    'source_id': source['id'], 
                    'title': source['title']
                })

        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")

    print("\nGenerating embeddings for all chunks...")
    embeddings = model.encode(
        all_chunks_for_chroma['documents'],
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # --- MODIFICATION START ---
    # Add to ChromaDB in batches to avoid exceeding the limit
    print("Adding embeddings to ChromaDB in batches...")
    BATCH_SIZE = 4000 # A safe batch size well below the limit
    total_chunks = len(all_chunks_for_chroma['ids'])

    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Adding to Chroma"):
        end_index = i + BATCH_SIZE
        
        collection.add(
            ids=all_chunks_for_chroma['ids'][i:end_index],
            embeddings=embeddings[i:end_index].tolist(),
            documents=all_chunks_for_chroma['documents'][i:end_index],
            metadatas=all_chunks_for_chroma['metadatas'][i:end_index]
        )
    # --- MODIFICATION END ---

    conn.commit()
    conn.close()
    print("\n🎉 Ingestion complete!")
    print(f"Total chunks processed: {total_chunks}")
if __name__ == "__main__":
    setup_database()
    process_and_embed()