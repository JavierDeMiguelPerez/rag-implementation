import os
from dotenv import load_dotenv

# --- IMPORTACIONES ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. CARGAR VARIABLES
load_dotenv()

# 2. CONFIGURAR MEMORIA
print("🧠 Cargando memoria vectorial...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

# 3. CONFIGURAR CEREBRO (ACTUALIZADO A LLAMA 3.3)
print("🤖 Conectando con el cerebro de Llama-3.3 via Groq...")
# Cambiamos el modelo viejo por el nuevo "versatile" que es muy potente
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 4. EL PROMPT
template = """
Eres Naval Ravikant. Usa el siguiente contexto para responder a la pregunta de forma sabia y directa.
Si no sabes la respuesta, di "No tengo esa información".

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# 5. CREAR LA CADENA
print("🔗 Creando la cadena de pensamiento...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# --- PRUEBA DE FUEGO ---
pregunta = ""
print(f"\n❓ Pregunta: {pregunta}")
print("🔍 Buscando en el libro y pensando...")

try:
    resultado = qa_chain.invoke({"query": pregunta})
    
    print("\n💬 RESPUESTA DE NAVAL BOT:")
    print("--------------------------------------------------")
    print(resultado["result"])
    print("--------------------------------------------------")

    print("\n📄 FUENTES UTILIZADAS:")
    for doc in resultado["source_documents"]:
        print(f"- {doc.page_content[:100]}...")
except Exception as e:
    print(f"\n❌ ERROR: {e}")