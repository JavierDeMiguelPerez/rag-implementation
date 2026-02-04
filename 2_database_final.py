import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. CARGAR VARIABLES
load_dotenv()

# Configuración
FILE_PATH = "El-Almanaque-de-Naval-Ravikant.pdf" 
CHROMA_PATH = "chroma_db"

def generar_db_final():
    print("Cargando PDF...")
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    
    # 2. EMBEDDINGS
    print("Cargando modelo de Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. CHUNKING "CLÁSICO" OPTIMIZADO
    # Estrategia: Chunks grandes (1500) para capturar contextos amplios (como listas).
    # Overlap alto (300) para asegurar que no cortamos una frase clave a la mitad.
    print("Cortando texto con estrategia 'Big Chunk'...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Aumentamos de 1000 a 1500 caracteres
        chunk_overlap=300 # Aumentamos el solapamiento
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Se han generado {len(chunks)} chunks robustos.")
    
    # 4. GUARDAR
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Base de datos fallida borrada.")

    print("Guardando la base de datos definitiva...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("¡Base de Datos Lista!")

if __name__ == "__main__":
    generar_db_final()