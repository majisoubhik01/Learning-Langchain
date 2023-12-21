from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

embeddings = HuggingFaceHubEmbeddings()
db = Chroma(
    persist_directory="emb", 
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

result = chain.run("What is an interesting fact about English language?")

print(result)