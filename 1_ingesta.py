import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN ---
FILE_PATH = "El-Almanaque-de-Naval-Ravikant.pdf"  # Asegúrate de que tu archivo se llame así o cambia esto

# 1. CARGAR EL PDF
# Usamos PyPDFLoader, que lee el archivo página por página
print(f"🔄 Cargando el archivo: {FILE_PATH}...")
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()

print(f"✅ PDF cargado. Total de páginas: {len(docs)}")

# 2. SPLITTING (FRAGMENTACIÓN)
# RecursiveCharacterTextSplitter es el estándar. 
# Intenta cortar por párrafos, luego por frases, luego por palabras.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Tamaño objetivo de cada trozo (caracteres)
    chunk_overlap=200   # Cuánto se repite del trozo anterior (para no perder contexto)
)

print("Dividiendo el texto en chunks...")
splits = text_splitter.split_documents(docs)

# --- RESULTADOS ---
print(f"Se han generado {len(splits)} chunks (fragmentos).")
print("\n--- EJEMPLO DE UN CHUNK (El primero) ---")
print(splits[0].page_content)
print("-----------------------------------------")