# Context-Aware RAG Architecture

A modular, production-oriented implementation of a **Retrieval-Augmented Generation (RAG)** system designed to interrogate long-form technical documents (PDFs) while maintaining conversational context.
This architecture is content-agnostic and can be adapted to any technical, legal, or corporate documentation by swapping the source file. It leverages **History-Aware Retrieval** to handle follow-up questions effectively.

<img width="717" height="548" alt="image" src="https://github.com/user-attachments/assets/7db26d10-de70-4799-8ae3-001febad9991" />


---

## Key Features

* **Strict RAG Implementation:** Minimizes hallucinations by strictly grounding LLM responses in vector-retrieved context. If the information isn't in the docs, the model admits ignorance.
* **Context-Aware Memory:** Solves the "vague follow-up" problem (e.g., *"How do I apply that?"*) by employing an intermediate LLM step to reformulate queries based on chat history before hitting the vector database.
* **Robust Ingestion Pipeline:** Uses an optimized `RecursiveCharacterTextSplitter` strategy with significant overlap to preserve context in lists, tables, and complex arguments.
* **Modern Tech Stack:** Built on Python 3.11+, LangChain v0.3, ChromaDB, and the latest Llama 3.3 (via Groq) for low-latency inference.

## Tech Stack

* **Orchestration:** LangChain (Community & Core)
* **LLM Inference:** Llama 3.3 70B Versatile (via Groq API)
* **Vector Store:** ChromaDB (Local persistence)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
* **Frontend:** Streamlit

## Project Structure

* `ingest.py`: **ETL Pipeline**. Handles document loading, text splitting, embedding generation, and vector store persistence.
* `app.py`: **Inference Engine**. A Streamlit application managing session state, chat history, and the RAG retrieval chain.

## Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

### 3. Configuration
Create a .env file in the root directory and add your Groq API key:
```python
GROQ_API_KEY=gsk_your_api_key_here
```

### 4. Customization (Optional)
To use your own data:
1. Replace the PDF file in the root directory.
2. Update the FILE_PATH constant in ingest.py.
3. (Optional) Update the system prompts in app.py to match the tone of your new document.

### 5. Execution
Step 1: Ingest Data Generate the vector embeddings (run this once or whenever the PDF changes).
```bash
python ingest.py
```

Step 2: Launch Application Start the chat interface.
```bash
streamlit run app.py
```

---
Developed as a Proof of Concept (PoC) for modern RAG architectures.
