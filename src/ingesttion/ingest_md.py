import logging
import sys
import os
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from llama_index.core.schema import TextNode
from uuid import uuid4
from llama_index.core import StorageContext
import json



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'pgvector_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MD_DIR = "D:\\Praveen\\AI\\Projects\\Assignments\\agentic-rag-assignment\\data\\processed\\md"

# Database configuration - simple password without special characters
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag_db"
DB_USER = "postgres"
DB_PASSWORD = "admin"

# Construct DATABASE_URL (no need for URL encoding now)
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Debug logging to verify the URL constructionprint(f"DEBUG: Constructed DATABASE_URL: {DATABASE_URL}")
print(f"DEBUG: DB_HOST: {DB_HOST}")
print(f"DEBUG: DB_PASSWORD (encoded): {quote_plus(DB_PASSWORD)}")

TABLE_NAME = "document_embeddings"
EMBED_DIM = 768

def validate_batch(batch, expected_dim=768):
    valid_nodes = []
    for node in batch:
        # Ensure node has text
        if not getattr(node, "text", None):
            print("❌ Node missing text, skipping")
            continue

        # Ensure embedding exists and has correct dimension
        emb = getattr(node, "embedding", None)
        if emb is None:
            print(f"⚠️ Node missing embedding, will generate")
            continue
        if len(emb) != expected_dim:
            print(f"❌ Node embedding dimension {len(emb)} != expected {expected_dim}, skipping")
            continue

        # Ensure metadata is JSON-serializable
        try:
            json.dumps(node.metadata or {})
        except Exception as e:
            print(f"❌ Node metadata not serializable: {e}, skipping")
            continue

        valid_nodes.append(node)
    
    print(f"✅ {len(valid_nodes)}/{len(batch)} nodes valid for insertion")
    return valid_nodes

def check_database_connection():
    """Test database connectivity and check for pgvector extension"""
    try:
        logger.info(f"Testing connection to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
        
        # Create engine for testing
        engine = create_engine(DATABASE_URL)
        
        # with engine.begin() as conn:
        #     conn.execute(
        #         text("INSERT INTO document_embeddings (text, metadata, embedding) VALUES ('TEST', '{}', ARRAY[0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.04529275,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607 ,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607,0.052667446,-0.012666334,0.016665852,0.005236814,-0.17998056,-0.045292765,0.032997377,-0.026936151,-0.026388718,-0.002536194,0.037798926,-0.003455602,0.007806842,0.02197607]::vector);")
        #     )

        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ PostgreSQL connected: {version}")
            
            # Check if pgvector extension exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """))
            has_pgvector = result.fetchone()[0]
            
            if not has_pgvector:
                logger.warning("⚠️ pgvector extension not found. Attempting to create...")
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    logger.info("✅ pgvector extension created successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to create pgvector extension: {e}")
                    logger.error("Run this SQL as superuser: CREATE EXTENSION vector;")
                    return False
            else:
                logger.info("✅ pgvector extension is installed")
            
            # Check if table exists
            result = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = '{TABLE_NAME}'
                );
            """))
            table_exists = result.fetchone()[0]
            
            if table_exists:
                logger.info(f"ℹ️ Table '{TABLE_NAME}' already exists")
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
                count = result.fetchone()[0]
                logger.info(f"ℹ️ Current row count: {count}")
                
                # Drop table if empty for fresh start
                # if count == 0:
                #     logger.info(f"Dropping empty table for fresh indexing...")
                #     conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
                #     conn.commit()
                #     logger.info(f"✅ Table dropped")
            else:
                logger.info(f"ℹ️ Table '{TABLE_NAME}' does not exist yet - will be created")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama embedding service"""
    try:
        logger.info("Testing Ollama embedding service...")
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url="http://localhost:11434",
        )
        
        # Test with a sample text
        test_embedding = embed_model.get_text_embedding("test")
        logger.info(f"✅ Ollama connected - embedding dim: {len(test_embedding)}")
        
        if len(test_embedding) != EMBED_DIM:
            logger.warning(f"⚠️ Embedding dimension mismatch! Expected {EMBED_DIM}, got {len(test_embedding)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        logger.error("And the model is pulled: ollama pull nomic-embed-text:v1.5")
        return False

def load_documents():
    """Load markdown documents with error handling"""
    try:
        logger.info(f"Loading documents from: {MD_DIR}")
        
        # Check if directory exists
        import os
        if not os.path.exists(MD_DIR):
            logger.error(f"❌ Directory does not exist: {MD_DIR}")
            return None
            
        # Check for markdown files
        md_files = []
        for root, dirs, files in os.walk(MD_DIR):
            md_files.extend([os.path.join(root, f) for f in files if f.endswith('.md')])
        
        logger.info(f"Found {len(md_files)} markdown files")
        
        if not md_files:
            logger.warning("⚠️ No markdown files found in directory")
            return None
        
        # Load documents
        reader = SimpleDirectoryReader(input_dir=MD_DIR, recursive=True)
        documents = reader.load_data()
        
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        # Log sample document info
        if documents:
            logger.debug(f"Sample document metadata: {documents[0].metadata}")
            logger.debug(f"Sample document text (first 200 chars): {documents[0].text[:200]}...")
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Failed to load documents: {e}")
        return None

def main():
    """Main indexing function with comprehensive error handling"""
    logger.info("="*60)
    logger.info("Starting PGVector Hybrid Indexing Process")
    logger.info("="*60)
    
    # Step 1: Check database connection
    if not check_database_connection():
        logger.error("Aborting: Database connection issues")
        return
    
    # Step 2: Check Ollama
    if not test_ollama_connection():
        logger.error("Aborting: Ollama connection issues")
        return
    
    # Step 3: Load documents
    documents = load_documents()
    if not documents:
        logger.error("Aborting: No documents to index")
        return
    
    try:
        # Step 4: Configure embedding model
        logger.info("Configuring embedding model...")
        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url="http://localhost:11434",
        )
        
        sample_embedding = Settings.embed_model.get_text_embedding("test")
        print(len(sample_embedding))

        # Step 5: Create vector store
        logger.info("Creating vector store...")
        
        # Use parsed components instead of URL to avoid re-encoding issues
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            table_name=TABLE_NAME,
            embed_dim=EMBED_DIM,
            hybrid_search=True,
            text_search_config="english",
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops"
            }
        )
        logger.info("✅ Vector store created")
        
        # Step 6: Parse documents into nodes first
        logger.info("Parsing documents into nodes...")
        md_parser = MarkdownNodeParser(include_metadata=True)
        nodes = md_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} nodes from markdown parsing")
        
        # Step 7: Further split nodes if needed
        logger.info("Splitting nodes into chunks...")
        text_splitter = TokenTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separator=" "
        )
        split_nodes = text_splitter.get_nodes_from_documents(nodes)
        clean_nodes = []
        for node in split_nodes:
            if "<!-- image -->" in node.text:
                continue
            clean_nodes.append(node)

        split_nodes = clean_nodes
        
        logger.info(f"Created {len(split_nodes)} chunks after splitting")

        # Step 8: Create index and add nodes
        logger.info("Creating vector store index and adding nodes...")
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="rag_db",
            user="postgres",
            password="admin"
        )

        register_vector(conn)
        cur = conn.cursor()
        
        # Add nodes to the index - use batch insertion for better performance
        logger.info("Inserting nodes into vector store...")
        batch_size = 100
        for i in range(0, len(split_nodes), batch_size):
            batch = split_nodes[i:i+batch_size]
            # batch = validate_batch(batch, expected_dim=EMBED_DIM)
            logger.info(f"Progress: {i}/{len(split_nodes)} nodes inserted")
            for node in batch:
                if not getattr(node, "embedding", None):
                    clean_text = node.text.replace("<!-- image -->", "").strip()
                    node.embedding = Settings.embed_model.get_text_embedding(clean_text)

            batch = [node for node in batch if len(node.embedding) == EMBED_DIM]

            batch_sql_values = []
            for node  in batch:
                vector_str = f"[{', '.join(str(x) for x in node.embedding)}]"
                json_str = json.dumps(node.metadata)
                batch_sql_values.append((node.text, json_str, vector_str))
          
            cur.executemany("""
                INSERT INTO document_embeddings (text, metadata, embedding)
                VALUES (%s, %s::jsonb, %s::vector)
                """, batch_sql_values)
        
        conn.commit()
        cur.close()
        conn.close()

        # ✅ Insert directly into Postgres via PGVectorStore
        # vector_store.add(batch)
        # index.insert_nodes(batch)
        
        logger.info(f"✅ All {len(split_nodes)} nodes inserted")
        
        # Step 9: Verify indexing
        logger.info("Verifying indexed data...")
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Check table structure
            result = conn.execute(text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{TABLE_NAME}'
                ORDER BY ordinal_position;
            """))
            columns = result.fetchall()
            logger.info("Table columns:")
            for col_name, col_type in columns:
                logger.info(f"  - {col_name}: {col_type}")
            
            # Check row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
            count = result.fetchone()[0]
            logger.info(f"✅ Final row count in table: {count}")
            
            # Check if embeddings are present
            result = conn.execute(text(f"""
                SELECT COUNT(*) 
                FROM {TABLE_NAME} 
                WHERE embedding IS NOT NULL
            """))
            embedding_count = result.fetchone()[0]
            logger.info(f"✅ Rows with embeddings: {embedding_count}")
            
            # Sample a row to verify (without array_length for vector type)
            result = conn.execute(text(f"""
                SELECT id, 
                       CASE WHEN text IS NOT NULL THEN SUBSTRING(text, 1, 100) ELSE NULL END as text_sample,
                       metadata::text as metadata_sample,
                       embedding IS NOT NULL as has_embedding
                FROM {TABLE_NAME} 
                LIMIT 1
            """))
            sample = result.fetchone()
            if sample:
                logger.info("Sample row:")
                logger.info(f"  - ID: {sample[0]}")
                logger.info(f"  - Text sample: {sample[1]}...")
                logger.info(f"  - Has embedding: {sample[3]}")
                
                # Get vector dimension using vector_dims function if available
                try:
                    result = conn.execute(text(f"""
                        SELECT vector_dims(embedding) as dims
                        FROM {TABLE_NAME} 
                        WHERE embedding IS NOT NULL
                        LIMIT 1
                    """))
                    dims = result.fetchone()
                    if dims:
                        logger.info(f"  - Embedding dimension: {dims[0]}")
                except:
                    # If vector_dims doesn't work, just skip
                    pass
        
        logger.info("="*60)
        logger.info(f"✅ Indexing complete! {len(split_nodes)} chunks → table '{TABLE_NAME}'")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)