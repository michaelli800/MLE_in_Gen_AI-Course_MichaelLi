from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv()  # This will load variables from the .env file into the environment

# Now you can access the API key
api_key = os.getenv("OPENAI_API_KEY")


# 1) LOAD your resume & docs
resume = PyPDFLoader("./python-developer-resume-example.pdf").load()
extras = TextLoader("./portfolio_notes.txt").load()
docs = resume + extras

# 2) CHUNK
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) EMBED + INDEX
emb = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, emb)

# 4) MAKE RAG chain
llm = OpenAI(temperature=0)
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})
)

def ask_me(question):
    print(agent.run(question))

# example
if __name__=="__main__":
    ask_me("What Python projects have Giulia contributed?")
