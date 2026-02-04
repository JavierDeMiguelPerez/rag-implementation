import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORTACIONES AVANZADAS (NIVEL 2) ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Naval Ravikant Bot", page_icon="⚓")
st.title("Naval Bot: Memoria Contextual")
st.markdown("*Ahora entiendo el contexto. Pregúntame '¿Qué es X?' y luego '¿Cómo consigo eso?'.*")

# 2. CARGA DE RECURSOS
@st.cache_resource
def iniciar_rag():
    load_dotenv()
    
    print("Cargando componentes...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # --- PASO A: EL REFORMULADOR DE PREGUNTAS ---
    # Este prompt NO responde la pregunta. Solo la reescribe para que tenga sentido por sí sola.
    contextualize_q_system_prompt = """
    Dada una historia de chat y la última pregunta del usuario 
    (que podría hacer referencia al contexto del chat), formula una pregunta independiente 
    que pueda entenderse sin la historia del chat. 
    NO respondas a la pregunta, solo reformúlala si es necesario y devuélvela.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # Aquí inyectamos el historial
            ("human", "{input}"),
        ]
    )
    
    # Este componente usa el LLM para reescribir la query ANTES de buscar en Chroma
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        vector_db.as_retriever(search_kwargs={"k": 6}), 
        contextualize_q_prompt
    )

    # --- PASO B: EL GENERADOR DE RESPUESTAS (NAVAL) ---
    # Este es el prompt clásico que ya tenías
    qa_system_prompt = """
    Eres Naval Ravikant. Usa los siguientes fragmentos de contexto recuperados para responder a la pregunta.
    Si no sabes la respuesta, di que no lo sabes. Usa un máximo de tres oraciones y sé conciso.
    Mantén un tono filosófico y directo.
    
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # El historial también sirve para dar tono
            ("human", "{input}"),
        ]
    )
    
    # Cadena que procesa los documentos
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # --- PASO C: LA GRAN UNIFICACIÓN ---
    # Unimos: (Historial + Query -> Query Reformulada) -> (Búsqueda) -> (Respuesta)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Inicializar
try:
    rag_chain = iniciar_rag()
    st.success("Sistema Contextual Activado.", icon="✅")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# 3. GESTIÓN DEL HISTORIAL (SESSION STATE)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. PINTAR EL CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. LÓGICA DE INTERACCIÓN
if prompt := st.chat_input("Pregunta algo..."):
    
    # Guardar input usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- CONVERTIR HISTORIAL DE STREAMLIT A FORMATO LANGCHAIN ---
        # LangChain necesita objetos HumanMessage y AIMessage, no diccionarios
        chat_history_core = []
        for msg in st.session_state.messages[:-1]: # Excluimos el último (el actual)
            if msg["role"] == "user":
                chat_history_core.append(HumanMessage(content=msg["content"]))
            else:
                chat_history_core.append(AIMessage(content=msg["content"]))
        
        try:
            # Invocamos la cadena pasando el 'chat_history' explícitamente
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history_core
            })
            
            respuesta_texto = response["answer"]
            
            message_placeholder.markdown(respuesta_texto)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            
        except Exception as e:
            message_placeholder.markdown(f"Error: {e}")