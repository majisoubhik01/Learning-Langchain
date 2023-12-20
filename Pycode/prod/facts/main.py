from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=10,
    chunk_overlap=0
)

loader = TextLoader("c:/Users/soubhik.maji/OneDrive - Subex Limited/Desktop/Courses/Langchain GPT/Pycode/prod/facts/facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

for doc in docs:
    print(doc.page_content)
    print("\n")