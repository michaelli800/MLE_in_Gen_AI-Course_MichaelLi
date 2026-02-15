## GPU Version Embedding
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Load your documents
resume = PyPDFLoader("./python-developer-resume-example.pdf").load()
extras = TextLoader("./portfolio_notes.txt").load()
docs = resume + extras

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings with OpenAI (you can replace this later with local embedding models)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embedding_model)



# 4. Connect to your vLLM server running on port 8000 (via SSH tunnel or locally)
llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="not-needed",
    model_name="HuggingFaceH4/zephyr-7b-alpha"  # matches your vLLM launch
)

# 5. Retrieval QA chain
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 6. Ask a question
def ask_me(question):
    print(agent.run(question))

if __name__ == "__main__":
    ask_me("What Python projects has Giulia contributed?")
