import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# CORRECCIÓN: Usamos la librería community que ya tienes instalada
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker

# 1. CARGAR VARIABLES
load_dotenv()

# Configuración
FILE_PATH = "El-Almanaque-de-Naval-Ravikant.pdf" 
CHROMA_PATH = "chroma_db"

def generar_db_semantica():
    print("Cargando PDF...")
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    
    # 2. EMBEDDINGS
    print("Cargando modelo de Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. CHUNKING SEMÁNTICO
    print("Analizando el significado del texto (Semantic Chunking)...")
    
    # Visualización mental: Imagina que el sistema lee frase a frase y mide la "distancia"
    # entre ellas. Si el tema cambia bruscamente (percentile 90), hace un corte.
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Se han generado {len(chunks)} chunks semánticos.")
    
    # 4. GUARDAR
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Base de datos antigua borrada.")

    print("Guardando nueva DB inteligente...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("🎉 ¡Base de Datos Semántica lista!")

if __name__ == "__main__":
    generar_db_semantica()