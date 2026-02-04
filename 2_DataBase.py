import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACIÓN ---
FILE_PATH = "El-Almanaque-de-Naval-Ravikant.pdf" # Tu archivo real
CHROMA_PATH = "chroma_db" # Carpeta donde se guardará la base de datos

# 1. CARGA Y CHUNKING (Repetimos el proceso anterior)
print("Cargando y procesando el PDF...")
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# 2. SISTEMA DE EMBEDDINGS
# Usamos un modelo local ligero y gratuito. 
# La primera vez que lo ejecutes tardará un poco en descargarse.
print("Inicializando modelo de Embeddings (HuggingFace)...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. CREAR Y GUARDAR LA BASE DE DATOS VECTORIAL
# Si ya existe la carpeta, la borramos para empezar de cero (limpieza)
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

print("Creando base de datos vectorial en disco (esto puede tardar unos segundos)...")
# Aquí ocurre la magia: Texto -> Números -> Disco
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function, 
    persist_directory=CHROMA_PATH
)

print(f"¡Éxito! Base de datos guardada en la carpeta: {CHROMA_PATH}")