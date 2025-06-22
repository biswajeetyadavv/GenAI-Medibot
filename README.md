# GenAI-Medibot: Context-Aware Medical Assistant using LLMs
Medibot is a context-aware medical chatbot that leverages large language models (LLMs) with FAISS-based vector memory to provide relevant, data-driven responses. It allows persistent memory and conversational context management, designed for efficiency and adaptability in healthcare-oriented environments.

## Features
- Integrates LLMs with vector memory for enhanced contextual conversation.
- Utilizes FAISS for efficient semantic search on embedded medical documents.
- Modular design for memory creation, connection, and inference.

## Project Structure
- data/                         # Data directory for medical documents
- vectorstore/db_faiss/         # FAISS vector database files  
- connect_memory_with_llm.py    # Loads memory and connects with LLM for inference  
- create_memory_for_llm.py      # Embeds documents and creates FAISS vector store  
- medibot.py                    # Main execution script for chatbot  
- requirements.txt              # Required Python packages  
- Pipfile / Pipfile.lock        # Pipenv environment configuration  
- LICENSE  
- README.md

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
or
using pipenv:
```bash
pipenv install
pipenv shell
```
3. **Prepare the Data**
```bash
Place your medical documents (PDF or text files) inside the data/ folder
```

4. **Create Vector Memory**
   Run the following script to embed the documents and generate the FAISS vector store:
```bash
python create_memory_for_llm.py
```

5. **Launch the Chatbot**
Start the chatbot interface that uses the embedded memory and connects to the LLM:
```bash
python connect_memory_with_llm.py
```


## Tech Stack
- Python 3.8+
- FAISS – for efficient vector similarity search
- OpenAI / compatible LLMs – for natural language understanding
- LangChain (if used) – for chaining memory and language tools
- Streamlit or Flask (optional frontend integration)

##Notes
- Ensure API keys or access tokens (e.g., OpenAI keys) are managed securely and excluded via .gitignore.
- The project supports easy extension for custom documents, additional tools, and deployment options.

##License
This project is licensed under the MIT License. See the LICENSE file for more details.
```bash
You can now paste this directly into your `README.md` file — it's compact, professional, and well-structured for viewers on GitHub or recruiters checking your profile. Let me know if you want to add **screenshots**, a **demo section**, or **badges** (like Python version, license, or build status).
```
