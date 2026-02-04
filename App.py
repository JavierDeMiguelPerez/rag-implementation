import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain & Vector Store Imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import Runnable

# --- CONFIGURATION ---
PAGE_TITLE = "RAG Contextual Assistant"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0

# Load environment variables
load_dotenv()

# --- INITIALIZATION & SETUP ---
st.set_page_config(page_title=PAGE_TITLE)
st.title(f"{PAGE_TITLE}")
st.markdown(
    """
    Intelligent assistant with conversational memory. 
    **Loaded Knowledge Base:** *The Almanack of Naval Ravikant*.
    """
)

@st.cache_resource
def get_rag_chain() -> Runnable:
    """
    Initializes the RAG (Retrieval-Augmented Generation) chain.
    This function is cached to avoid reloading the model and embeddings on every interaction.

    Returns:
        Runnable: The compiled LangChain RAG pipeline.
    """
    
    # 1. API Key Validation
    if not os.getenv("GROQ_API_KEY"):
        st.error("Critical Error: GROQ_API_KEY not found in environment variables.")
        st.stop()

    # 2. Load Embeddings & Vector Database
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(CHROMA_PATH):
        st.error(f"Vector database not found at '{CHROMA_PATH}'. Please run 'ingest.py' first.")
        st.stop()
        
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Configure LLM
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    
    # 4. History Aware Retriever Setup
    # This prompts the LLM to rephrase the user's latest query using the chat history context
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        vector_db.as_retriever(search_kwargs={"k": 5}), 
        contextualize_q_prompt
    )

    # 5. Question Answering Chain Setup
    qa_system_prompt = (
        "You are an expert assistant for document analysis. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, clearly state that you do not have that information. "
        "Keep your answer professional, concise, and helpful.\n\n"
        "CONTEXT:\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 6. Final RAG Chain Construction
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- MAIN APPLICATION LOGIC ---

def main():
    # Initialize the RAG system
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize the application: {e}")
        st.stop()

    # Session State Management (Chat History)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input Handling
    if prompt := st.chat_input("Ask a question based on the document..."):
        
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("*Analyzing documents...*")
            
            # Convert session state to LangChain message format
            chat_history_objs: List[BaseMessage] = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history_objs.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_objs.append(AIMessage(content=msg["content"]))
            
            try:
                # Execute RAG Chain
                response = rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": chat_history_objs
                })
                
                full_response = response["answer"]
                
                # Display and Save Response
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                message_placeholder.error(f"An error occurred while generating the response: {e}")

if __name__ == "__main__":
    main()