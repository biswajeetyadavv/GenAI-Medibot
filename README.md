# GenAI-Medibot

# Medibot: Context-Aware Medical Assistant using LLMs
Medibot is a context-aware medical chatbot that leverages large language models (LLMs) with FAISS-based vector memory to provide relevant, data-driven responses. It allows persistent memory and conversational context management, designed for efficiency and adaptability in healthcare-oriented environments.

## Features
- Integrates LLMs with vector memory for enhanced contextual conversation.
- Utilizes FAISS for efficient semantic search on embedded medical documents.
- Modular design for memory creation, connection, and inference.

## Project Structure
├── data/                          # Data directory for medical documents  
├── vectorstore/db_faiss/         # FAISS vector database files  
├── connect_memory_with_llm.py    # Loads memory and connects with LLM for inference  
├── create_memory_for_llm.py      # Embeds documents and creates FAISS vector store  
├── medibot.py                    # Main execution script for chatbot  
├── requirements.txt              # Required Python packages  
├── Pipfile / Pipfile.lock        # Pipenv environment configuration  
├── LICENSE  
└── README.md

## Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/biswajeetyadavv/medibot.git
cd medibot
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```
