import os
import shutil
import logging
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure basic logging structure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Path to the source PDF file (Technical or Legal documents recommended)
FILE_PATH = "El-Almanaque-de-Naval-Ravikant.pdf" 
# Directory to store the persistent vector database
CHROMA_PATH = "chroma_db"
# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main() -> None:
    """
    Main execution pipeline:
    1. Validates input file.
    2. Loads and splits the PDF.
    3. Generates embeddings.
    4. Creates/Resets the Chroma vector database.
    """
    
    # 1. File Validation
    if not os.path.exists(FILE_PATH):
        logger.error(f"File not found: '{FILE_PATH}'. Please ensure the file exists in the root directory.")
        return

    try:
        # 2. Load PDF
        logger.info(f"Loading document: {FILE_PATH}...")
        loader = PyPDFLoader(FILE_PATH)
        docs = loader.load()
        
        # 3. Initialize Embeddings
        logger.info(f"Initializing embedding model ({EMBEDDING_MODEL_NAME})...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # 4. Text Splitting (Chunking)
        logger.info("Processing and splitting document...")
        
        # Using a larger chunk size (1500) to preserve context in lists and complex arguments.
        # The overlap (300) prevents context loss between chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Generated {len(chunks)} text fragments (chunks).")
        
        # 5. Create/Reset Vector Database
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            logger.info("Previous vector database cleared.")

        logger.info("Generating new ChromaDB vector store...")
        Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        
        logger.info("Ingestion completed successfully! You can now run the application.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during ingestion: {e}")
        raise e

if __name__ == "__main__":
    main()