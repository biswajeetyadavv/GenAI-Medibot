import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Setup page configuration
st.set_page_config(page_title="Medibot - AI Health Assistant", page_icon="🩺")

# Cache vector store to avoid reloading on every interaction
@st.cache_resource
def load_vectorstore():
    """Loads FAISS vectorstore using HuggingFace sentence transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Define the custom prompt template
def get_prompt_template() -> PromptTemplate:
    template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know — don't try to make it up.
Only use information from the given context.

Context:
{context}

Question:
{question}

Provide a direct, concise answer. Avoid unnecessary text.
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load LLM from Hugging Face Inference API
def load_llm():
    """Initializes and returns the language model from Hugging Face."""
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

# Streamlit UI and logic
def main():
    st.title("🩺 Medibot - Your AI Medical Assistant")
    st.markdown("Ask any health-related question. The assistant will answer based on its knowledge base.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Take user input
    user_input = st.chat_input("Ask your medical question here...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("💡 Medibot is thinking..."):
            try:
                # Load components
                vectorstore = load_vectorstore()
                llm = load_llm()

                # Setup Retrieval QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": get_prompt_template()}
                )

                # Run the chain
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]
                sources = response.get("source_documents", [])

                # Format source info (optional)
                # if sources:
                #     source_texts = "\n".join(
                #         f"- {doc.metadata.get('source', 'Unknown source')}" for doc in sources
                #     )
                #     answer += f"\n\n**Sources:**\n{source_texts}"

                # Display response
                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Something went wrong:\n\n`{str(e)}`")

if __name__ == "__main__":
    main()
