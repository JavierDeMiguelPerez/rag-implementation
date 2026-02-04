# ⚓ Naval Bot: Asistente Filosófico RAG

Un sistema de **Retrieval-Augmented Generation (RAG)** capaz de responder preguntas filosóficas basándose exclusivamente en el libro *"El Almanaque de Naval Ravikant"*, evitando alucinaciones y manteniendo el contexto de la conversación.

## 🚀 Arquitectura Técnica

Este proyecto implementa un pipeline de datos completo:

* **Ingesta de Datos:** Procesamiento de PDF con particionado optimizado (Recursive Character Splitter) para maximizar la recuperación de contexto.
* **Vector Store:** Uso de **ChromaDB** para almacenamiento de embeddings generados con `sentence-transformers/all-MiniLM-L6-v2`.
* **Cerebro (LLM):** Integración con **Llama 3.3 (70b)** vía Groq API para inferencia de ultra-baja latencia.
* **Memoria Conversacional:** Implementación de un *History-Aware Retriever* que reformula las preguntas del usuario basándose en el historial del chat.
* **Interfaz:** Frontend interactivo construido con **Streamlit**.

## 🛠️ Instalación y Uso

1. **Clonar el repositorio:**
    git clone [URL_DE_TU_REPO]
    cd naval_bot
2. **Instalar dependencias:**
    python -m venv venv
    source venv/bin/activate  # O venv\Scripts\activate en Windows
    pip install -r requirements.txt
3. **Configurar entorno: Crea un archivo .env y añade tu API Key de Groq:**
    GROQ_API_KEY=gsk_...
4. **Generar la Base de Datos Vectorial:**
    python 2_database_final.py
5. **Lanzar la App:**
    streamlit run RagBot.py

## 🧠 Retos Superados
 - Optimización de estrategias de Chunking (comparativa entre Semantic Chunking vs Fixed-size) para mejorar la recuperación de listas y conceptos largos.

 - Gestión de estado de sesión en Streamlit para mantener la coherencia del chat.